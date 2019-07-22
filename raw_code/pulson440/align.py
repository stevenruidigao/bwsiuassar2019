#imports pandas
import pandas
import numpy as np
import pickle
with open('Mandrill_1way_Misaligned1_data.pkl', 'rb') as f:
    data=pickle.load(f)
import matplotlib.pyplot as plt

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
    perfect_thing = newnew[0][0]
    print(data['motion_timestamps'][perfect_thing])


data_align(data['scan_data'], data['platform_pos'], data['range_bins'], data['scan_timestamps'], data['motion_timestamps'], data['corner_reflector_pos'])



#panda_stuff('ExampleMotionCapture1.csv')
