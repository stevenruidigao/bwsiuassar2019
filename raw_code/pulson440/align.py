#imports desired libs
import pandas
import numpy as np
import pickle
import matplotlib.pyplot as plt

#opens and loads pickle file
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

#finds the time stamp at which the drone takes off relative to the motion capture system's timer
def MC_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    new = np.where(one_way_range==first_value, 0, one_way_range)
    newnew = np.nonzero(new)
    tkf_scan_num = newnew[0][0]
    return data['motion_timestamps'][tkf_scan_num]
##    print("Time of takeoff after the motion capture device was turned on (seconds): ", tkf_motion_timestamp)

#function within a function; finds the time stamp at which the drone takes off relative to the radar's timer
def RD_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos):
    print(np.linalg.norm(scan_data - scan_data[0], axis=1))
    print(np.where(np.linalg.norm(scan_data - scan_data[0], axis=1) > 4.5, 0, 1))
    return data['scan_timestamps'][np.where(np.linalg.norm(scan_data - scan_data[0], axis=1) > 4.5, 1, 0).nonzero()[0][0]] - data['scan_timestamps'][0]


#data align function (incomplete, nothing is there we just called our functions here so the data align function doesnt throw an error)
def data_align(scan_data, platform_pos, range_bins, scan_timestamps, motion_timestamps, corner_reflector_pos):
    a = MC_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos)
    b = RD_tkf_timestamp(scan_data, platform_pos, range_bins, corner_reflector_pos)
##    print(scan_timestamps, motion_timestamps)
    print(a, b)
    aligned_mocap_times = motion_timestamps + b - a
    print(aligned_mocap_times)
data_align(data['scan_data'], data['platform_pos'], data['range_bins'], data['scan_timestamps'], data['motion_timestamps'], data['corner_reflector_pos'])


#plt.imshow()

#panda_stuff('ExampleMotionCapture1.csv')
