import pickle
import numpy as np
import matplotlib.pyplot as plt

def backprojection(data):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"]
    xlen = 240
    ylen = 240
    return_data = np.zeros((xlen, ylen))
    pixel_x = -1
    for x in np.linspace(-3, 3, xlen):
        pixel_x += 1
        pixel_y = -1 
        for y in np.linspace(-3, 3, ylen):
            pixel_y += 1
            for scan_number in range(len(platform_pos)):
                pos = platform_pos[scan_number]
                distance = (np.sum(np.square(np.array([y, x, 0]) - pos))) ** (1 / 2)
                range_bin = (np.abs(range_bins - distance)).argmin()
                return_data[pixel_x, pixel_y] += scan_data[scan_number][range_bin]
            return_data[pixel_x, pixel_y] = abs(return_data[pixel_x, pixel_y])
    return return_data

with open('Mandrill_1way_data.pkl', 'rb') as f:
    data = pickle.load(f)
##    print(data)

bpdat = backprojection(data)
print(bpdat)
plt.imshow(bpdat)
plt.pause(60)
