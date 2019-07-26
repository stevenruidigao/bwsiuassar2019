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

def backprojection2dfast(data, start=-3, stop=3, resolution=0.05, twodimbins=False):
    """Runs backprjection on SAR data
    
    Args:
        data (dict)
            Data dictionary loaded from data file.
            
        start (int)
            The x and y coordinates to start the image at.
            
        stop (int)
            The x and y coordinates to stop the image at.
            
        resolution (float)
            The number of pixels per square unit."

        twodimbins (bool)
            Whether or not to use weird two dimensional range bins.
    """
    # Load the data
    scan_data = data["pulses"]
    print(scan_data.shape)
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"][0] if twodimbins else data["range_bins"]
    # The number of pixels per square unit
    alen = int((stop - start) / resolution) + 1
    print(alen)
    xlen = alen
    ylen = alen
    # Create the return data array with datatype complex128 
    return_data = np.zeros((xlen, ylen), dtype=np.complex128)
    # Loop over each scan and add the appropriate intensities to each pixel through np.interp
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number] # Take the position of the radar at the time of the current scan
##        print(np.linspace(start, stop, xlen))
##        input("")
        meshgrid = np.asarray(np.meshgrid(np.linspace(start, stop, xlen), np.linspace(start, stop, ylen))) # Create a 2D grid
        points = np.concatenate((meshgrid, np.zeros((1, xlen, ylen)))).transpose(1, 2, 0) # Add a Z-dimesion and fill it with zeros
        distances = np.linalg.norm(points - pos, axis=2) # Take the distance of each coordinate from the radar
        interp = np.interp(distances, range_bins, scan_data[scan_number]).reshape(xlen, ylen) # Interpolate on those distances using range_bin data and scan_data
        return_data += np.flipud(interp) # Flip the returned intensities and add the intensities to the return_data array
    return np.abs(return_data)

# Get arguments
parser = argparse.ArgumentParser(description=" - Runs backprojection on SAR data.")
parser.add_argument("--filename", "-f", type=str, required=True, help=" - The SAR data filename")
parser.add_argument("--start", type=float, default=-3, help=" - The start of the coordinate plane")
parser.add_argument("--stop", type=float, default=3, help=" - The end of the coordinate plane")
parser.add_argument("--resolution", "--res", type=float, default=0.05, help=" - The resolution of the SAR image")
parser.add_argument("--two_dimensional_range_bins", "--2D-range-bins",  "--2D_bins", action="store_true", help=" - Enables use of weird 2D range_bins.")
##parser.add_argument("--mode", "-m", type=str, default="2dfast", help=" - The mode to run back projection in.")
args = parser.parse_args()

with open(args.filename, 'rb') as f:
    data = pickle.load(f) # Load data from file

bpdat = backprojection2dfast(data, args.start,args.stop, args.resolution, args.two_dimensional_range_bins) # Run backprojection
plt.imshow(bpdat) # Show the graph
plt.show() # Prevent the graph from instantly disappearing
