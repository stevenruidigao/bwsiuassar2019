import pickle
import numpy as np
import matplotlib.pyplot as plt

def backprojection(data):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"]
##    print(range_bins)
    return_data = np.zeros((scan_data.shape[0], scan_data.shape[1]))
##    print(scan_data.shape[0], platform_pos.shape[0])
    pixel_x = -1
    for x in np.linspace(-3, 3, scan_data.shape[0]):
        pixel_x += 1
        pixel_y = -1 
        for y in np.linspace(-3, 3, scan_data.shape[1]):
            pixel_y += 1
            for scan_number in range(len(platform_pos)):
                pos = platform_pos[scan_number]
                distance = (np.sum(np.square(np.array([x, y, 0]) - pos))) ** (1 / 2)
##                print(distance)
                range_bin = min(range(len(range_bins)), key=lambda i: abs(range_bins[i] - distance))
##                range_bin = -1
##                for i in range(len(range_bins[0])):
##                    range_bin_distance = range_bins[0, i]
####                    print(range_bin_distance)
##                    if range_bin_distance >= distance:
##                        range_bin = i
####                    print(i)
####            print(range_bin)
####            print(scan_number)
####            print(x, y)
            return_data[pixel_x, pixel_y] += np.abs(scan_data[scan_number][range_bin])
    return return_data

with open('2Points_1way_data.pkl', 'rb') as f:
    data = pickle.load(f)
##    print(data)

bpdat = backprojection(data)
print(bpdat)
plt.imshow(bpdat)
plt.pause(60)
