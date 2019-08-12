import pickle
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

if Path("..//").resolve().as_posix() not in sys.path:
    sys.path.insert(0, Path("..//").resolve().as_posix())
from common.helper_functions import replace_nan


corner = 'Corner Reflector.csv'
motion = 'FinalTestDay_1.csv'
radar = 'FinalTestDay_1'


corner_data = pd.read_csv(corner, header=2)
motion_data = pd.read_csv(motion, header=2)

with open(radar, 'rb') as f:
    radar_data = pickle.load(f)


def combine_data(motion_data, radar_data, corner_reflector):
    data = radar_data.copy()
    data['scan_timestamps'] = data['timestamps']
    data['motion_timestamps'] = motion_data['motion_timestamps']
    data['platform_pos'] = motion_data['platform_pos']
    data['corner_reflector_pos'] = corner_reflector
    return data


def get_cr_pos(corner, name):
    start_index = -1
    curr = 0
    for col in corner.columns:
        if col == name:
            start_index = curr
            break
        curr += 1
    start_index += 4
    ret = (float(corner.iloc[3, start_index]),
           float(corner.iloc[3, start_index + 1]),
           float(corner.iloc[3, start_index + 2]))
    return ret


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
    ret['motion_timestamps'] = np.asarray(motion.iloc[3:, 1]).astype(np.float)
    # ret['motion_timestamps'].index = range(len(ret['motion_timestamps']))
    ret['platform_pos'] = np.column_stack((motion.iloc[3:, start_index],
                                           motion.iloc[3:, start_index + 1],
                                           motion.iloc[3:, start_index + 2]))
    ret['platform_pos'] = ret['platform_pos'].astype(np.float)
    # ret['platform_pos'].index = range(len(ret['platform_pos']))
    return ret


def MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos):
    max = len(platform_pos)
    rng = range(max)
    for i in rng:
        if i == 0:
            continue
        if np.sum(np.abs(platform_pos[i] - platform_pos[0])) > 0.5:
            return i
    return 0


def MC_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos):
    max = len(platform_pos)
    rng = range(max)
    for i in reversed(rng):
        if i == 0:
            continue
        if np.sum(np.abs(platform_pos[i] - platform_pos[max - 1])) > 0.5:
            return i
    return 0


# function within a function; finds the time stamp at which the drone takes off relative to the radar's timer
def RD_tkf_indexn(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - (corner_reflector_pos[0])), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    return (np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > 1000).nonzero()[0][0]


def RD_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    return (np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > 5).nonzero()[0][-1]


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
        # if not count % 10 == 0:
            # continue
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
        results = np.reshape(np.interp(temp, range_bins, scan_data[count]), (int((xstop-xstart)/resolution),
                                                                             int((ystop-ystart)/resolution)))
        ret += results
        count += 1

    # Display the value
    # plt.imshow(np.abs(ret))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.imshow(np.abs(ret),
               extent=(
                   xstart,
                   xstop,
                   ystop,
                   ystart))
    plt.colorbar()
    plt.show()
    plt.pause(5)
    return np.abs(ret)


def motion_align(data):
    scan_data = data['scan_data']
    platform_pos = data['platform_pos']
    range_bins = data['range_bins']
    corner_reflector_pos = data['corner_reflector_pos']
    motion_timestamps = data['motion_timestamps']
    scan_timestamps = data['scan_timestamps']

    newdata = data.copy()

    newdata['scan_timestamps'] -= newdata['scan_timestamps'][0]

    pos_x = newdata['platform_pos'][:, 0]
    pos_y = newdata['platform_pos'][:, 1]
    pos_z = newdata['platform_pos'][:, 2]

    temp = np.arange(motion_timestamps[0], motion_timestamps[-1],
                     np.abs(scan_timestamps[1] - scan_timestamps[0]) / 1000)

    realigned_pos_x = np.interp(temp, newdata['motion_timestamps'], pos_x)
    realigned_pos_y = np.interp(temp, newdata['motion_timestamps'], pos_y)
    realigned_pos_z = np.interp(temp, newdata['motion_timestamps'], pos_z)

    newdata['platform_pos'] = np.column_stack((realigned_pos_x, realigned_pos_y, realigned_pos_z))
    newdata['scan_timestamps'] = newdata['scan_timestamps'] / 1000
    newdata['motion_timestamps'] = temp
    return newdata


def replace_nans(data):
    newdata = data.copy()
    replacement_data = replace_nan(newdata['motion_timestamps'], newdata['platform_pos'])
    newdata['motion_timestamps'] = np.asarray(replacement_data[0])
    newdata['platform_pos'] = np.asarray(replacement_data[1])
    return newdata


def cropper(data, *ranges):
    scan_data = data['scan_data']
    platform_pos = data['platform_pos']
    range_bins = data['range_bins']
    corner_reflector_pos = data['corner_reflector_pos']
    motion_timestamps = data['motion_timestamps']
    scan_timestamps = data['scan_timestamps']

    mc_takeoff = MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos)
    mc_landing = MC_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos)

    # rd_takeoff = 0
    rd_takeoff = RD_tkf_indexn(scan_data, platform_pos, range_bins, corner_reflector_pos)

    # rd_landing = len(scan_data)
    rd_landing = RD_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos)

    if len(ranges) > 1:
        start_index = (np.abs(range_bins - ranges[0])).argmin()
        end_index = (np.abs(range_bins - ranges[1])).argmin()
        scan_data = scan_data[rd_takeoff:rd_landing, start_index:end_index]
        range_bins = range_bins[start_index:end_index]
    else:
        scan_data = scan_data[rd_takeoff:rd_landing]
        range_bins = range_bins[0]

    scan_timestamps = scan_timestamps[rd_takeoff:rd_landing]

    platform_pos = platform_pos[mc_takeoff:mc_landing]
    motion_timestamps = motion_timestamps[mc_takeoff:mc_landing]

    data['scan_timestamps'] = scan_timestamps
    data['scan_data'] = scan_data

    data['platform_pos'] = platform_pos
    motion_timestamps -= motion_timestamps[0]
    motion_timestamps += 15.2
    data['motion_timestamps'] = motion_timestamps

    range_bins -= 0.5
    data['range_bins'] = range_bins

    return data


fig = plt.figure()
ax = fig.gca(projection='3d')
motion_data = get_pos(motion_data, 'Radar')
corner = [[-3.94, 0.36, 0.63], [1.59, 0.22, -1.85]]
data = combine_data(motion_data, radar_data, corner)


data = replace_nans(data)

ax.plot(data['platform_pos'][:, 0], data['platform_pos'][:, 1], data['platform_pos'][:, 2], '-*', label='Radar')
plt.show()

data = motion_align(data)

data = cropper(data, 1, 20)

better_back_projection(data, 0.01, -5, -2, -1, 2)

plt.imshow(np.abs(data['scan_data']),
           extent=(
                   data['range_bins'][0],
                   data['range_bins'][-1],
                   data['scan_timestamps'][-1] - data['scan_timestamps'][0],
                   0),
           aspect='auto')

plt.xlabel('Range (m)')
plt.ylabel('Elapsed Time (s)')


r1 = np.sqrt(np.sum(
                   (data['platform_pos'] - data['corner_reflector_pos'][0])**2, 1))
plt.plot(r1, data['motion_timestamps'], 'r--', label='Corner Reflector 1')
plt.show()
