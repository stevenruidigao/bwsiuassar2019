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
##import common.helper_functions as helper
from pathlib import Path
import sys
if Path("..//").resolve().as_posix() not in sys.path:
    sys.path.insert(0, Path("..//").resolve().as_posix())
from common.helper_functions import replace_nan

def backprojection(data, xstart=-3, xstop=3, ystart=-3, ystop=3, zstart=-3, zstop=3, xresolution=0.05, yresolution=0.05, zresolution=0.05, mode="2dfast"):
    if mode == "2dfast":
        return backprojection2dfast(data, xstart, xstop, ystart, ystop, xresolution, yresolution)
    elif mode =="3dfast":
        return backprojection3dfast(data, xstart, xstop, ystart, ystop, zstart, zstop, xresolution, yresolution, zresolution)
    elif mode =="2dslow":
        return backprojection2dslow(data, xstart, xstop, ystart, ystop, xresolution, yresolution)
    else:
        raise RuntimeError("Invalid mode!")

def backprojection2dfast(data, xstart=-3, xstop=3, ystart=-3, ystop=3, xresolution=0.05, yresolution=0.05):
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
    """
    # Load the data
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"]
    # The number of pixels per dimension
##    alen = int((stop - start) / resolution) + 1
    xlen = int((xstop - xstart) / xresolution) + 1
    ylen = int((ystop - ystart) / yresolution) + 1
    # Create the return data array with datatype complex128 
    return_data = np.zeros((xlen, ylen), dtype=np.complex128)
##    print(return_data)
    # Loop over each scan and add the appropriate intensities to each pixel through np.interp
    for scan_number in range(1):
        pos = platform_pos[scan_number] # Take the position of the radar at the time of the current scan
        meshgrid = np.asarray(np.meshgrid(np.linspace(xstart, xstop, xlen), np.linspace(ystart, ystop, ylen))) # Create a 2D grid
##        print(meshgrid[0].shape, meshgrid[1].shape, np.zeros((1, ylen, xlen)).shape)
        points = np.concatenate((meshgrid, np.zeros((1, ylen, xlen)))).transpose(1, 2, 0) # Add a Z-dimesion and fill it with zeros
        distances = np.linalg.norm(points - pos, axis=2) # Take the distance of each coordinate from the radar
##        print(distances)
        interp = np.interp(distances, range_bins, scan_data[scan_number]).reshape(xlen, ylen) # Interpolate on those distances using range_bin data and scan_data
        return_data += interp # Flip the returned intensities and add the intensities to the return_data array
##        print(npinterp)
##    print(return_data)
    return np.abs(return_data)

def backprojection3dfast(data, xstart=-3, xstop=3, ystart=-3, ystop=3, zstart=-3, zstop=3, xresolution=0.05, yresolution=0.05, zresolution=0.05):
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
    """
    # Load the data
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"]
    # The number of pixels per dimension
##    alen = int((stop - start) / resolution) + 1
    xlen = int((xstop - xstart) / xresolution) + 1
    ylen = int((ystop - ystart) / yresolution) + 1
    zlen = int((zstop - zstart) / zresolution) + 1
    return_data = np.zeros((xlen, ylen, zlen), dtype=np.complex128)
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number]
        meshgrid = np.meshgrid(np.linspace(xstart, xstop, xlen), np.linspace(ystart, ystop, ylen), np.linspace(zstart, zstop, zlen))
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

def backprojection2dslow(data, xstart=-3, xstop=3, ystart=-3, ystop=3, xresolution=0.05, yresolution=0.05):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"]
##    print(range_bins)
##    alen = int((stop - start) / resolution) + 1
    xlen = int((xstop - xstart) / xresolution) + 1
    ylen = int((ystop - ystart) / yresolution) + 1
    return_data = np.zeros((xlen, ylen))
    pixel_x = -1
    for x in np.linspace(xstart, xstop, xlen):
        pixel_x += 1
        pixel_y = -1 
        for y in np.linspace(ystart, ystop, ylen):
            pixel_y += 1
            for scan_number in range(len(platform_pos)):
                pos = platform_pos[scan_number]
                distance = (np.sum(np.square(np.array([y, -x, 0]) - pos))) ** (1 / 2)
                range_bin = (np.abs(range_bins - distance)).argmin()
                return_data[pixel_x, pixel_y] += scan_data[scan_number][range_bin]
            return_data[pixel_x, pixel_y] = abs(return_data[pixel_x, pixel_y])
    return return_data

def MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold=0.5):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
##    print(one_way_range == first_value)
    indices = (one_way_range != first_value).nonzero()
    # print(indices)
    tkf_scan_num = indices[0][0]
    return tkf_scan_num + 1


# function within a function; finds the time stamp at which the drone takes off relative to the radar's timer
def RD_tkf_index_old(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
    for k in range(1, num_scans):
        current = scan_data[k, cr_first_rbin]
        previous = scan_data[k-1, cr_first_rbin]
        if np.abs(current - previous) > 0.5:
            return k

def RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold=0.5):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
    return (np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > threshold).nonzero()[0][0] # 11.208849086510625 

def RD_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold=0.5):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
    ##    np.set_printoptions(threshold=np.inf)
    ##    print(np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]))
    ##    np.set_printoptions(threshold=1000)
    return (np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > threshold).nonzero()[0][-1]

def motion_align(data, threshold=0.5, shift=0):
    scan_data = data['scan_data']
    platform_pos = data['platform_pos']
    range_bins = data['range_bins']
    corner_reflector_pos = data['corner_reflector_pos']
    motion_timestamps = data['motion_timestamps']
    scan_timestamps = data['scan_timestamps'] - data['scan_timestamps'][0]
    
    RD_change_time = scan_timestamps[RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold)]
    MC_change_time = motion_timestamps[MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold)]
    
    print(RD_change_time, MC_change_time)
    
    newdata = data.copy()
    
    newdata['scan_timestamps'] -= newdata['scan_timestamps'][0]
    newdata['motion_timestamps'] += RD_change_time - MC_change_time + shift
        
    pos_x = newdata['platform_pos'][:, 0]
    pos_y = newdata['platform_pos'][:, 1]
    pos_z = newdata['platform_pos'][:, 2]

    realigned_pos_x = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_x)
    realigned_pos_y = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_y)
    realigned_pos_z = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_z)
    
    newdata['platform_pos'] = np.column_stack((realigned_pos_x, realigned_pos_y, realigned_pos_z))
    newdata['motion_timestamps'] = newdata['scan_timestamps']
    return newdata

def range_norm(data):
    scan_data = data['scan_data']
    range_bins = data['range_bins']
    norm_scan_data = np.copy(scan_data)
    norm_scan_data *= ((range_bins / range_bins[0]) ** 4)
    return norm_scan_data

def replace_nans(data):
    newdata = data.copy()
    replacement_data = replace_nan(newdata['motion_timestamps'], newdata['platform_pos'])
    newdata['motion_timestamps'] = replacement_data[0]
    newdata['platform_pos'] = replacement_data[1]
    return newdata

def crop(data, start_range, end_range, threshold=0.5):
##    data = olddata.copy()
    scan_data = data['scan_data']
    platform_pos = data['platform_pos']
    range_bins = data['range_bins']
    corner_reflector_pos = data['corner_reflector_pos']
    motion_timestamps = data['motion_timestamps']
    scan_timestamps = data['scan_timestamps']

    mid = int(len(scan_timestamps) / 2)

    rd_takeoff = RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold)

    rd_landing = RD_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold)

##    print(type(start_range))
##    print(type(range_bins))
##    print(range_bins - start_range)
##    print(range_bins)
    
    start_index = np.abs(range_bins - start_range).argmin()
    end_index = np.abs(range_bins - end_range).argmin()

    scan_data = scan_data[rd_takeoff:rd_landing, start_index:end_index]
    scan_timestamps = scan_timestamps[rd_takeoff:rd_landing]

    platform_pos = platform_pos[rd_takeoff:rd_landing]
    motion_timestamps = motion_timestamps[rd_takeoff:rd_landing]

    range_bins = range_bins[start_index:end_index]

    data['scan_timestamps'] = scan_timestamps
    data['scan_data'] = scan_data

    data['platform_pos'] = platform_pos
    data['motion_timestamps'] = motion_timestamps

    data['range_bins'] = range_bins

    return data

def get_pos_data(csv_data, name):
    return_data = tuple()
    start_index = -1
    curr = 0
    for col in csv_data.columns:
        if col == name:
            start_index = curr
            break
        curr += 1
    start_index += 4
    
    x = csv_data.iloc[3:, start_index]
    y = csv_data.iloc[3:, start_index + 1]
    z = csv_data.iloc[3:, start_index + 2]
    motion_timestamps = csv_data.iloc[3:, 1]
    
    x.index = range(len(x))
    y.index = range(len(y))
    z.index = range(len(z))
    motion_timestamps.index = range(len(motion_timestamps))
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    motion_timestamps = np.asarray(motion_timestamps)
    
    return_data[0] = np.stack((x, y, z), 1)
    return_data[1] = motion_timestamps
    
    return return_data

parser = argparse.ArgumentParser(description=" - Runs backprojection on SAR data.")
parser.add_argument("--filename", "-f", type=str, required=True, help=" - The SAR data filename.")
parser.add_argument("--xstart", type=float, default=-3, help=" - The start of the x-axis.")
parser.add_argument("--xstop", type=float, default=3, help=" - The end of the x-axis.")
parser.add_argument("--ystart", type=float, default=-3, help=" - The start of the y-axis.")
parser.add_argument("--ystop", type=float, default=3, help=" - The end of the y-axis.")
parser.add_argument("--zstart", type=float, default=-3, help=" - The start of the z-axis.")
parser.add_argument("--zstop", type=float, default=3, help=" - The end of the z-axis.")
parser.add_argument("--xresolution", "--xres", type=float, default=0.05, help=" - The resolution of the SAR image.")
parser.add_argument("--yresolution", "--yres", type=float, default=0.05, help=" - The resolution of the SAR image.")
parser.add_argument("--zresolution", "--zres", type=float, default=0.05, help=" - The resolution of the SAR image.")
parser.add_argument("--two_dimensional_range_bins", "--2D-range-bins",  "--2D_bins", action="store_true", help=" - Enables use of weird 2D range_bins.")
parser.add_argument("--mode", "-m", type=str, default="2dfast", help=" - The mode to run back projection in.")
parser.add_argument("--realign", "--align", action="store_true", help=" - Realigns the motion capture data and the radar data.")
parser.add_argument("--realign_thresh", "--thresh", type=float, default=0.5, help=" - Realigns the motion capture data and the radar data with this threshold.")
parser.add_argument("--realign_shift", "--shift", type=float, default=0, help=" - Amount to shift motion capture data by when aligning.")
parser.add_argument("--crop", "-c", nargs=2, metavar=("start", "stop"), help=" - Crops the corner reflectors out using start and stop positions.")
parser.add_argument("--normalize", "--norm", action="store_true", help=" - Uses the range_norm function to account for different ranges in the list of intesities")
parser.add_argument("--clim", nargs=2, help=" - Color map stuff")
parser.add_argument("--reflectors", type=int, default=1, help=" - Number of corner reflectors")
parser.add_argument("--replace-nans", "--nans", action="store_true", help = "Whether or not to replace NaNs in motion capture data.")
args = parser.parse_args()

print(args)

with open(args.filename, 'rb') as f:
    data = pickle.load(f)


##print(data)

if args.replace_nans:
    data = replace_nans(data)

if args.two_dimensional_range_bins:
    data['range_bins'] = data['range_bins'][0]
    
if args.realign:
    data = motion_align(data, args.realign_thresh, args.realign_shift)
    plt.imshow(np.abs(data['scan_data']),
        extent=(
            data['range_bins'][0],
            data['range_bins'][-1],
            data['scan_timestamps'][-1] - data['scan_timestamps'][0],
            0))

    plt.xlabel('Range (m)')
    plt.ylabel('Elapsed Time (s)')

    for refnum in range(args.reflectors):
        r1 = np.sqrt(np.sum(
                (data['platform_pos'] - data['corner_reflector_pos'][refnum, :])**2, 1))
        plt.plot(r1, data['motion_timestamps'], 'r--', label='Corner Reflector' + str(refnum + 1))
    plt.show()

print(data["platform_pos"], data["platform_pos"].shape)

if args.crop is not None:
    data = crop(data, float(args.crop[0]), float(args.crop[1]), args.realign_thresh)

if args.normalize:
    data = range_norm(data)

##print(data)

bpdat = backprojection(data, args.xstart, args.xstop, args.ystart, args.ystop, args.zstart, args.zstop, args.xresolution, args.yresolution, args.zresolution, args.mode)
##print(bpdat)
##print(bpdat.shape)
print(data["platform_pos"], data["platform_pos"].shape)

plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.imshow(bpdat,
    extent=(
        args.xstart,
        args.xstop,
        args.ystop,
        args.ystart))
##plt.clim(100)
plt.colorbar()
plt.plot(data['platform_pos'][:, 0], data['platform_pos'][:, 1], 'r-')
plt.show()
