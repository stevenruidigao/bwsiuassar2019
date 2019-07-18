#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A script to run backprojection."""

__author__ = "Steven Gao"
__version__ = "1.0"
__maintainer__ = "Steven Gao"
__email__ = "stevenruidigao@gmail.com"
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def backprojection(data, start=-3, stop=3, resolution=0.05, twodimbins=False, mode="2dfast"):
    if mode == "2dfast":
        return backprojection2dfast(data, start, stop, resolution, twodimbins)
    elif mode =="3dfast":
        return backprojection3dfast(data, start, stop, resolution, twodimbins)
    elif mode =="2dslow":
        return backprojection2dslow(data, start, stop, resolution, twodimbins)
    else:
        raise RuntimeError("Invalid mode!")

def backprojection2dfast(data, start=-3, stop=3, resolution=0.05, twodimbins=False):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"][0] if twodimbins else data["range_bins"]
    alen = int((stop - start) / resolution)
    xlen = alen
    ylen = alen
    return_data = np.zeros((xlen, ylen), dtype=np.complex128)
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number]
        meshgrid = np.asarray(np.meshgrid(np.linspace(start, stop, xlen), np.linspace(start, stop, ylen)))
        points = np.concatenate((meshgrid, np.zeros((1, xlen, ylen)))).transpose(1, 2, 0)
        distances = np.linalg.norm(points - pos, axis=2)
        infer = np.reshape(distances, distances.size)
        interp = np.interp(infer, range_bins, scan_data[scan_number]).reshape(xlen, ylen)
        return_data += np.flipud(interp)
    return np.abs(return_data)

def backprojection3dfast(data, start=-3, stop=3, resolution=0.05, twodimbins=False):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"][0] if twodimbins else data["range_bins"]
    alen = int((stop - start) / resolution)
    xlen = alen
    ylen = alen
    zlen = alen
    return_data = np.zeros((xlen, ylen, zlen), dtype=np.complex128)
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number]
        meshgrid = np.meshgrid(np.linspace(start, stop, xlen), np.linspace(start, stop, ylen), np.linspace(start, stop, zlen))
        points = np.stack(meshgrid).transpose(1, 2, 3, 0)
        distances = np.linalg.norm(points - pos, axis=3)
        infer = np.reshape(distances, distances.size)
        interp = np.interp(infer, range_bins, scan_data[scan_number]).reshape(xlen, ylen, zlen)
        return_data += np.flipud(interp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(return_data[:,0],return_data[:,1],return_data[:,2])
    plt.show()
    return np.abs(return_data)

def backprojection2dslow(data, start=-3, stop=3, resolution=0.05, twodimbins=False):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"][0] if twodimbins else data["range_bins"]
    print(range_bins)
    alen = int((stop - start) / resolution)
    xlen = alen
    ylen = alen
    return_data = np.zeros((xlen, ylen))
    pixel_x = -1
    for x in np.linspace(start, stop, xlen):
        pixel_x += 1
        pixel_y = -1 
        for y in np.linspace(start, stop, ylen):
            pixel_y += 1
            for scan_number in range(len(platform_pos)):
                pos = platform_pos[scan_number]
                distance = (np.sum(np.square(np.array([y, -x, 0]) - pos))) ** (1 / 2)
                range_bin = (np.abs(range_bins - distance)).argmin()
                return_data[pixel_x, pixel_y] += scan_data[scan_number][range_bin]
            return_data[pixel_x, pixel_y] = abs(return_data[pixel_x, pixel_y])
    return return_data

parser = argparse.ArgumentParser(description=" - Runs backprojection on SAR data.")
parser.add_argument("--filename", "-f", type=str, required=True, help=" - The SAR data filename")
parser.add_argument("--start", type=float, default=-3, help=" - The start of the coordinate plane")
parser.add_argument("--stop", type=float, default=3, help=" - The end of the coordinate plane")
parser.add_argument("--resolution", "--res", type=float, default=0.05, help=" - The resolution of the SAR image")
parser.add_argument("--two_dimensional_range_bins", "--2D-range-bins",  "--2D_bins", action="store_true", help=" - Enables use of weird 2D range_bins.")
parser.add_argument("--mode", "-m", type=str, default="2dfast", help=" - The mode to run back projection in.")
args = parser.parse_args()

with open(args.filename, 'rb') as f:
    data = pickle.load(f)

print(args)

bpdat = backprojection(data, args.start,args.stop, args.resolution, args.two_dimensional_range_bins, args.mode)
plt.imshow(bpdat)
plt.show()
