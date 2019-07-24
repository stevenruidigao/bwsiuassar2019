# imports desired libs
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# opens and loads pickle file
with open('Mandrill_1way_Misaligned1_data.pkl', 'rb') as f:
    data = pickle.load(f)

"""
h_fig_1=plt.figure()
h_fig_2=plt.figure()
plt.axes()
plt.imshow(x, extent=(left, right, bottom, top))
#left right is origin to end of x axis, opposite for bottom top and y axis
plt.imshow(np.abs(data['scan_data']), extent=(data['range_bins'][0,0], data['range_bins'][0,-1], data['scan_timestamps']
    [-1]-data['scan_timestamps'][0], 0))
plt.xlabel('Range (m)')
plt.ylabel('Elapsed Time (s)')
"""


# reads the CSV file
def panda_stuff(path):
    data = pd.read_csv(path, header=2)
    return data


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
        if np.abs(current - previous) > 4.5:
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


data_align(data['scan_data'], data['platform_pos'], data['range_bins'], data['scan_timestamps'],
           data['motion_timestamps'], data['corner_reflector_pos'])


#plt.imshow()

#panda_stuff('ExampleMotionCapture1.csv')
