import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

def backprojection3d(data, start=-3, stop=3, resolution=0.05):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"]
    alen = int((stop - start) / resolution)
    xlen = alen
    ylen = alen
    zlen = alen
    return_data = np.zeros((xlen, ylen))
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number]
        meshgrid = np.meshgrid(np.linspace(start, stop, xlen), np.linspace(start, stop, ylen), np.linspace(start, stop, zlen))
        print(meshgrid[0].shape)
        points = np.dstack(meshgrid)
        print(points.shape)
    return np.abs(return_data)

def backprojection2d(data, start=-3, stop=3, resolution=0.05):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"]
##    print(range_bins.shape, scan_data[0].shape)
    alen = int((stop - start) / resolution)
    xlen = alen
    ylen = alen
    return_data = np.zeros((xlen, ylen), dtype=np.complex128)#, dtype=np.dtype.complex128)
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number]
        
        meshgrid = np.asarray(np.meshgrid(np.linspace(start, stop, xlen), np.linspace(start, stop, ylen)))
##        print(meshgrid.shape)
        points = np.concatenate((meshgrid, np.zeros((1, xlen, ylen)))).transpose(1, 2, 0)
##        print(points.shape)
        distances = np.linalg.norm(points - pos, axis=2)
##        print(distances.shape)
        infer = np.reshape(distances, distances.size)
        interp = np.interp(infer, range_bins, scan_data[scan_number]).reshape(xlen, ylen)
        return_data += np.flipud(interp)
        
##        pixel_x = -1
##        for x in np.linspace(start, stop, xlen):
##            pixel_x += 1
##            pixel_y = -1 
##            for y in np.linspace(start, stop, ylen):
##                pixel_y += 1
##                distance = (np.sum(np.square(np.array([y, -x, 0]) - pos))) ** (1 / 2)
##                range_bin = (np.abs(range_bins - distance)).argmin()
##                return_data[pixel_x, pixel_y] += scan_data[scan_number][range_bin]
    return np.abs(return_data)

parser = argparse.ArgumentParser(description="Run backprojection on SAR data.")
parser.add_argument("--filename", "-f", type=str, required=True, help="SAR data filename")
parser.add_argument("--start", type=float, default=-3, help="Start of the coordinate plane")
parser.add_argument("--stop", type=float, default=3, help="End of the coordinate plane")
parser.add_argument("--resolution", "--res", type=float, default=0.05, help="Resolution of the SAR image")
args = parser.parse_args()

with open(args.filename, 'rb') as f:
    data = pickle.load(f)
##    print(data)

bpdat = backprojection2d(data, args.start, args.stop, args.resolution)
plt.imshow(bpdat)
plt.show()
