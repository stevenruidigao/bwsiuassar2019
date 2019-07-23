#imports pandas
import pandas
import numpy as np
import pickle
import matplotlib.pyplot as plt
with open('Mandrill_1way_Misaligned1_data.pkl', 'rb') as f:
    data=pickle.load(f)

"""
h_fig_1=plt.figure()
h_fig_2=plt.figure()
plt.axes()
plt.imshow(x, extent=(left, right, bottom, top))
#left right is origin to end of x axis, opposite for bottom top and y axis
plt.imshow(np.abs(data['scan_data']), extent=(data['range_bins'][0,0], data['range_bins'][0,-1], data['scan_timestamps'][-1]-data['scan_timestamps'][0], 0))
plt.xlabel('Range (m)')
plt.ylabel('Elapsed Time (s)')
"""

#reads the CSV file
def panda_stuff(x):
    stuff=pandas.read_csv(x)
    print(stuff)
    return stuff


def data_align(scan_data, platform_pos, range_bins, scan_timestamps, motion_timestamps, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    new = np.where(one_way_range==first_value, 0, one_way_range)
    newnew = np.nonzero(new)
    tkf_scan_num = newnew[0][0]
    tkf_motion_timestamp = data['motion_timestamps'][tkf_scan_num]
    print("Time of takeoff after the motion capture device was turned on (seconds): ", tkf_motion_timestamp)
    def find_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos):
        one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
        first_value = one_way_range[0]
        cr_first_rbin = np.argmin(np.abs(first_value-range_bins))
        num_scans = len(scan_data)
        for k in range(1, num_scans):
            yeet = scan_data[k, cr_first_rbin]
            yeet_two = scan_data[k-1, cr_first_rbin]
            if np.abs(yeet - yeet_two) > 4.5:
                tkf_scan_timestamp = data['scan_timestamps'][k]
                print("Time of takeoff after the radar was turned on (seconds): ", tkf_scan_timestamp)
                return k


    find_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos)

data_align(data['scan_data'], data['platform_pos'], data['range_bins'], data['scan_timestamps'], data['motion_timestamps'], data['corner_reflector_pos'])


#plt.imshow()

#panda_stuff('ExampleMotionCapture1.csv')
