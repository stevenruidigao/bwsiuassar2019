import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
corner_block_path = ''
motion_data_path = ''
'''
filepath = '/Users/ryanberry/data/sar_data/simulated/Mandrill_1way_Misaligned1_data.pkl'

'''
corner_block_data = pd.read_csv(corner_block_path, header=2)
motion_data = pd.read_csv(motion_data_path, header=2)
'''
with open(filepath, 'rb') as f:
    data = pickle.load(f)

# print(data)


def get_pos(motion, name):
    start_index = -1
    curr = 0
    for col in motion.columns:
        if col == name:
            start_index = curr
            break
        curr += 1
    start_index += 4
    ret = dict()
    ret['Time (Seconds)'] = motion.iloc[3:, 1]
    ret['Time (Seconds)'].index = range(len(ret['Time (Seconds)']))
    ret['X'] = motion.iloc[3:, start_index]
    ret['X'].index = range(len(ret['X']))
    ret['Y'] = motion.iloc[3:, start_index + 1]
    ret['Y'].index = range(len(ret['Y']))
    ret['Z'] = motion.iloc[3:, start_index + 2]
    ret['Z'].index = range(len(ret['Z']))
    return ret


def MC_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    range_with_zeros = np.where(one_way_range == first_value, 0, one_way_range)
    indices = np.nonzero(range_with_zeros)
    # print(indices)
    tkf_scan_num = indices[0][0]
    return tkf_scan_num


# function within a function; finds the time stamp at which the drone takes off relative to the radar's timer
def RD_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos, scan_timestamps):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
    for k in range(1, num_scans):
        current = scan_data[k, cr_first_rbin]
        previous = scan_data[k-1, cr_first_rbin]
        if np.abs(current - previous) > 0.5:
            return k


# data align function (incomplete, nothing is there we just called our
# functions here so the data align function doesnt throw an error)
def data_align(scan_data, platform_pos, range_bins, scan_timestamps, motion_timestamps, corner_reflector_pos):
    motion_change_time = MC_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos)
    radar_change_time = RD_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos, scan_timestamps)
    # print(data['scan_timestamps'])
    real_scan_times = scan_timestamps - scan_timestamps[0]
    # print(real_scan_times)
    tkf_motion_timestamp = motion_timestamps[motion_change_time]
    tkf_scan_timestamp = real_scan_times[radar_change_time]
    aligned_motion_times = motion_timestamps + (tkf_scan_timestamp - tkf_motion_timestamp)
    # print(tkf_scan_timestamp, tkf_motion_timestamp)
    # print(aligned_motion_times)
    return aligned_motion_times


def motion_change(scan_data, platform_pos, range_bins, scan_timestamps, motion_timestamps, corner_reflector_pos):
    return MC_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos)


def better_back_projection(data, resolution, xstart, xstop, ystart, ystop):
    fig = plt.figure()

    # Import the separate lists of data from the pickle file
    scan_data = data['scan_data']
    platform_pos = data['platform_pos']  # Create variable with all the platform positions
    platform_pos = np.asarray(platform_pos)
    range_bins = data['range_bins']  # Create variable with all of the range bins

    ret = np.zeros((int((xstop-xstart)/resolution), int((ystop-ystart)/resolution)), dtype=np.complex128)

    # Create variables (all of them are np arrays) that represent the x, y and z axis
    possible_x = np.linspace(xstart, xstop,
                             num=int((xstop-xstart)/resolution))
    possible_y = np.linspace(ystart, ystop,
                             num=int((ystop-ystart)/resolution))
    z_layer = np.zeros((int((xstop-xstart)/resolution), int((ystop-ystart)/resolution)))

    # Create an array of all possible points: Three layers, x, y, and z.  z is always zero.
    points = np.meshgrid(possible_x, possible_y)
    points = np.asarray(points)
    points = np.stack((points[0], -points[1], z_layer))

    count = 0
    for pos in platform_pos:
        # Create another array of the same size, this time with the current position of the platform
        pos_x = [pos[0]] * int((xstop-xstart)/resolution)
        pos_y = [pos[1]] * int((ystop-ystart)/resolution)
        pos_z = np.zeros((int((xstop-xstart)/resolution), int((ystop-ystart)/resolution)))
        pos_z.fill(pos[2])
        posit = np.meshgrid(pos_x, pos_y)
        posit = np.asarray(posit)
        posit = np.stack((posit[0], posit[1], pos_z))

        # Create another array of the same size, with each value being the range from that point to the platform
        ranges = np.linalg.norm(np.subtract(points, posit), axis=0)

        # Add the value at the range for each point to the running total of values
        temp = ranges.flatten()
        results = np.reshape(np.interp(temp, range_bins[0], scan_data[count]), (int((xstop-xstart)/resolution),
                                                                                int((ystop-ystart)/resolution)))
        ret += results
        count += 1

    # Display the value
    plt.imshow(np.abs(ret))
    plt.show()
    plt.pause(5)
    return np.abs(ret)


def realign_position(data):
    pos_x = data['platform_pos'][:, 0]
    pos_y = data['platform_pos'][:, 1]
    pos_z = data['platform_pos'][:, 2]

    realigned_pos_x = np.interp(data['scan_timestamps'], data['motion_timestamps'], pos_x)
    realigned_pos_y = np.interp(data['scan_timestamps'], data['motion_timestamps'], pos_y)
    realigned_pos_z = np.interp(data['scan_timestamps'], data['motion_timestamps'], pos_z)
    
    data['platform_pos'] = np.column_stack((realigned_pos_x, realigned_pos_y, realigned_pos_z))
    return data


def get_ranges(data):



# diction = get_pos(data, "LB_Marker")

data['motion_timestamps'] = data_align(data['scan_data'], data['platform_pos'], data['range_bins'],
                                       data['scan_timestamps'], data['motion_timestamps'], data['corner_reflector_pos'])

data = realign_position(data)

# data['motion_timestamps'] = np.linspace(data['motion_timestamps'][0], data['motion_timestamps'][-1],
#                                         len(data['range_bins'][0]))

# data['motion_timestamps'] = np.interp(data['motion_timestamps'], data['range_bins'][0], data['motion_timestamps'])
# better_back_projection(data, 0.01, -3, 3, -3, 3)


# print(diction['Time (Seconds)'])
# print('\n')
# print(diction['X'])
# print('\n')
# print(diction['Y'])
# print('\n')
# print(diction['Z'])

# extent=(left, right, bottom, top) - Changing axis, left right are min max X, bot top are min max Y.

plt.imshow(np.abs(data['scan_data']),
           extent=(
                   data['range_bins'][0, 0],
                   data['range_bins'][0, -1],
                   data['scan_timestamps'][-1] - data['scan_timestamps'][0],
                   0))
                   
plt.xlabel('Range (m)')
plt.ylabel('Elapsed Time (s)')
plt.pause(5)

print(len(data['scan_data']) == len(data['platform_pos']))

ranges = get_ranges(data)
plt.plot(data['platform_pos'], np.abs(data['scan_data']), 'r--')
plt.pause(5)


