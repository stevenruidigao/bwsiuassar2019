'''
COMMENT PROGRAM DESCRIPTION HERE
'''

#SOMEONE WRITE DOCUMENTATION FOR ARGUMENTS 


# imports desired libs
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# opens and loads pickle file
with open('Mandrill_1way_Misaligned3_data.pkl', 'rb') as f:
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


#reads the CSV file
def panda_stuff(path):
    data = pd.read_csv(path, header=2)
    return data

#defines a function in which the motion capture system determines at what scan number of the radar the one way range between the corner
#reflector and radar changes
def MC_change_index(scan_data, platform_pos, range_bins, corner_reflector_pos):

    #calculates the one way range between the radar and the corner reflector at each scan, stores all ranges for each corresponding scan to
    #a numpy array to be indexed later
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))

    #indexes the first value in numpy array of one way ranges, called one_way_range; first value is more than likely to be identical to
    #subsequent values because the range will not change for a few scans until the radar is moved. thus, it is sufficient to index only the
    #first range value
    first_value = one_way_range[0]

    #we can use first value to determine when the range changes by using numpy.where; it will compare the first value with every single value
    #in one_way_range. if the first value is equal to a value in one_way_range (or if the first argument evaluates to TRUE), it will take the
    #second argument, 0, and add it to a numpy array called range_with_zeros. if FALSE, it will take the respective one_way_range element and
    #add it to the array. this is done so any nonzero values can be distinguished as changes in the range between the corner reflector and
    #radar.
    range_with_zeros = np.where(one_way_range == first_value, 0, one_way_range)

    #returns scan number (index number) of all nonzero elements in range_with_zeros and adds it to numpy array 'indices'
    indices = np.nonzero(range_with_zeros)

    #assigns the first value of indices to tfk_scan_num because first element of the array is the scan number (index number) at which the radar
    #first takes off, or when the range first changes
    tkf_scan_num = indices[0][0]

    #returns the radar's takeoff scan number to be used later to determine which parts of our matplotlip image is necessary or not
    return tkf_scan_num


#defines a function to determine at what scan number of the radar the one way range between the corner reflector and radar changes
def RD_change_index(scan_data, platform_pos, range_bins, corner_reflector_pos, scan_timestamps):

    #calculates the one way range between the radar and the corner reflector at each scan, stores all ranges for each corresponding scan to
    #a numpy array to be indexed later
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))

    #indexes the first value in numpy array of one way ranges, called one_way_range; first value is more than likely to be identical to
    #subsequent values because the range will not change for a few scans until the radar is moved. thus, it is sufficient to index only the
    #first range value
    first_value = one_way_range[0]

    #COMMENT HERE
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))

    #takes the length of scan data to determine how many scans were done, assigned to num_scans
    num_scans = len(scan_data)

    #print("THIS IS SCAN DATA: ", scan_data)
    #print("THIS IS MOTION CAPTURE DAYDU: ", platform_pos)

    #COMMENT HERE
    for k in range(1, num_scans):
        current = scan_data[k, cr_first_rbin]
        previous = scan_data[k - 1, cr_first_rbin]
        if np.abs(current - previous) > 2.5:
            return k

#COMMENTING OF FUNCTION UNFINISHED

# data align function (incomplete, nothing is there we just called our
# functions here so the data align function doesnt throw an error)
def data_align(scan_data, platform_pos, range_bins, scan_timestamps, motion_timestamps, corner_reflector_pos):

    motion_change_time = MC_change_index(scan_data, platform_pos, range_bins, corner_reflector_pos)

    radar_change_time = RD_change_index(scan_data, platform_pos, range_bins, corner_reflector_pos, scan_timestamps)

    print("Radar change time here: ", radar_change_time)
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
