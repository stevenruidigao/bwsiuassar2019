import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('Mandrill_1way_data.pkl', 'rb') as f:
    data = pickle.load(f)


def back_projection(data, resolution, start, stop):
    scan_data = data['scan_data']  # Create variable with all the scan data
    platform_pos = data['platform_pos']  # Create variable with all the platform positions
    range_bins = data['range_bins']  # Create variable with all of the range bins
    possible_x = np.linspace(start, stop, num=int((stop-start)/resolution))  # Create a list with M values between -3 and 3
    possible_y = np.linspace(start, stop, num=int((stop-start)/resolution))  # Create a list with N values between -3 and 3

    return_data = np.zeros((int((stop-start)/resolution), int((stop-start)/resolution)))  # Create an M x N matrix to return

    xpix = 0
    for x in possible_x:
        ypix = 0
        for y in possible_y:
            point = 0
            scan = 0
            for pos in platform_pos:
                rng = np.sum(np.square(pos - (-y, -x, 0)))**0.5
                ind = (np.abs(range_bins - rng)).argmin()
                point += scan_data[scan][ind]
                scan += 1
            return_data[xpix, ypix] = np.abs(point)
            ypix += 1
        xpix += 1
    plt.imshow(return_data)
    print("Pausing...")
    plt.pause(10)
    return return_data


back_projection(data, 0.05, -3, 3)





