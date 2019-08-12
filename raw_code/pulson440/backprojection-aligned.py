#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A script to run backprojection."""

__author__ = "Steven Gao"
__version__ = "1.0"
__maintainer__ = "Steven Gao"
__email__ = "stevenruidigao@gmail.com"
import pickle
import numpy as np
import pandas as pd
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
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number] # Take the position of the radar at the time of the current scan
        meshgrid = np.asarray(np.meshgrid(np.linspace(xstart, xstop, xlen), np.linspace(ystart, ystop, ylen))) # Create a 2D grid
##        print(meshgrid[0].shape, meshgrid[1].shape, np.zeros((1, ylen, xlen)).shape)
        points = np.concatenate((meshgrid, np.zeros((1, ylen, xlen)))).transpose(1, 2, 0) # Add a Z-dimesion and fill it with zeros
        distances = np.linalg.norm(points - pos, axis=2) # Take the distance of each coordinate from the radar
##        print(np.sum(points - pos, axis=3))
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
##    print(np.abs(one_way_range - first_value))
    indices = (one_way_range != first_value).nonzero()
##    indices = (np.abs(one_way_range - first_value) > threshold).nonzero()
##    print(indices)
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
##    np.set_printoptions(threshold=np.inf)
##    print(np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]))
##    np.set_printoptions(threshold=1000)
##    return (np.abs(np.linalg.norm(scan_data - scan_data[0], axis=1)) > threshold).nonzero()[0][-1]
##    print((np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > threshold).nonzero()[0][0])
    plt.figure()
##    print(scan_data[0])
    plt.plot(scan_data[:, cr_first_rbin - 1], 'r--')
    plt.show()
    plt.figure()
    return (np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > threshold).nonzero()[0][0]
##    return (np.abs(np.linalg.norm(scan_data - scan_data[0], axis=1)) > threshold).nonzero()[0][0] # 11.208849086510625 

def RD_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold=0.5):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
    ##    np.set_printoptions(threshold=np.inf)
    ##    print(np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]))
    ##    np.set_printoptions(threshold=1000)
##    return (np.abs(np.linalg.norm(scan_data - scan_data[0], axis=1)) > threshold).nonzero()[0][-1]
##    print(np.abs(scan_data[1400]))
    return (np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > threshold).nonzero()[0][-1]

def time_align(data, threshold=0.5, shift=0):
    scan_data = data['scan_data']
    platform_pos = data['platform_pos']
    range_bins = data['range_bins']
    corner_reflector_pos = data['corner_reflector_pos']
    motion_timestamps = data['motion_timestamps']
    scan_timestamps = data['scan_timestamps'] - data['scan_timestamps'][0]
    
    RD_change_time = scan_timestamps[RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold)]
    MC_change_time = motion_timestamps[MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold)]
    
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

def range_align(data, distance):
    newdata = data.copy()
    newdata['range_bins'] += distance
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
##    print(newdata)
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

##    rd_takeoff = mid - 30 #
##    print(RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold))
    rd_takeoff = RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold)
##    print(RD_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold))
##    rd_landing = mid + 30 #
    rd_landing = RD_lnd_index(scan_data, platform_pos, range_bins, corner_reflector_pos, threshold)

    print(rd_takeoff, rd_landing)
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

def combine_data(motion_data, radar_data, corner_reflector):
    data = radar_data.copy()
    data['scan_timestamps'] = data['timestamps'] / 1000
    data['motion_timestamps'] = motion_data['motion_timestamps']
    data['platform_pos'] = motion_data['platform_pos']
    data['corner_reflector_pos'] = np.asarray([corner_reflector])
    return data


def get_cr_pos(corner, name):
    start_index = -1
    curr = 0
    for col in corner.columns:
        if col == name:
            start_index = curr
            break
        curr += 1
    start_index += 4
    ret = np.asarray([float(corner.iloc[3, start_index]),
           float(corner.iloc[3, start_index + 1]),
           float(corner.iloc[3, start_index + 2])])
    return ret


def get_pos_data(motion, name):
    start_index = -1
    curr = 0
    for col in motion.columns:
        if col == name:
            start_index = curr
            break
        curr += 1
    start_index += 4
    ret = dict()
    ret['motion_timestamps'] = np.asarray(motion.iloc[3:, 1]).astype(np.float)
    # ret['motion_timestamps'].index = range(len(ret['motion_timestamps']))
    ret['platform_pos'] = np.column_stack((motion.iloc[3:, start_index],
                                           motion.iloc[3:, start_index + 1],
                                           motion.iloc[3:, start_index + 2]))
    ret['platform_pos'] = ret['platform_pos'].astype(np.float)
    # ret['platform_pos'].index = range(len(ret['platform_pos']))
    return ret

def display_backprojected_image(backprojected_img, x_axis_bounds, y_axis_bounds, png_filename, 
                                x_label='X (m)', y_label='Y (m)', title='Backprojected Image', 
                                aspect='equal'):
    """Display and save backprojected image with labels.
    
    Args:
        backprojected_img (numpy.ndarray)
            Matrix containing backprojected image. Assumes that rows correspond to y-axis and 
            columns correspond to x-axis.
            
        x_axis_bounds (tuple)
            First element is the minimum/leftmost/first column's x-value while second element is the 
            maximum/rightmost/last column's x-value. These should be in meters.
    
        y_axis_bounds (tuple)
            First element is the minimum/bottom/last row's y-value while second element is the 
            maximum/top/first row's y-value. These should be in meters.
            
        png_filename (str)
            Path and name of PNG file of image to save. Must include PNG extension.
            
        x_label (string)
            X-axis label. Defaults to 'X (m)'.
            
        y_label (string)
            Y-axis label. Defaults to 'Y (m)'
            
        title (string)
            Title of image. Defaults to 'Backprojected Image'.
            
        aspect (string)
            Aspect ratio of displayed image. This should be any of the options supported by
            the 'aspect' keyword argument of matplotlib.pyplot.imshow 
            (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html). Defaults to 
            'equal' so that pixels present their true aspect ratio.
    """
    # Display the backprojected image
    hFig = plt.figure()
    hAx = plt.subplot(111)
    hImg = hAx.imshow(backprojected_img, extent=x_axis_bounds + y_axis_bounds)
    hAx.set_aspect(aspect=aspect)
    hAx.set_xlabel(x_label)
    hAx.set_ylabel(y_label)
    hAx.set_title(title)
    hFig.colorbar(hImg)
    
    # maximize window before showing and saving
    # TODO: This may need to change depending on you matplotlib backend; refer to this post for 
    # some potential solutions, 
    # https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen/32428266
##    figManager = plt.get_current_fig_manager()
##    figManager.window.showMaximized()
    plt.show()
    
    # Save the iamge
    hFig.savefig(png_filename)
    
def save_data_pickle(backprojected_img, x_axis, y_axis, pkl_filename):
    """Save data into pickle.
    
    Args:
        backprojected_img (numpy.ndarray)
            Matrix containing backprojected image. Assumes that rows correspond to y-axis and 
            columns correspond to x-axis.
            
        x_axis (numpy.ndarray)
            The x-coordinates of each column in the backprojected image. These should be in meters.
    
        y_axis (tuple)
            The y-coordinates of each row in the backprojected image. These should be in meters.
            
        pkl_filename (str)
            Path and name of pickle file to save.
    """
    # Open and save pickle
    with open(pkl_filename, 'wb') as f:
        data = {'backprojected_img': backprojected_img,
                'x_axis': x_axis,
                'y_axis': y_axis}
        pickle.dump(data, f)

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
parser.add_argument("--two_dimensional_range_bins", "--2D_range_bins",  "--2D_bins", action="store_true", help=" - Enables use of weird 2D range_bins.")
parser.add_argument("--mode", "-m", type=str, default="2dfast", help=" - The mode to run back projection in.")
parser.add_argument("--realign", "--align", action="store_true", help=" - Realigns the motion capture data and the radar data.")
parser.add_argument("--realign_thresh", "--thresh", type=float, default=0.5, help=" - Realigns the motion capture data and the radar data with this threshold.")
parser.add_argument("--realign_shift", "--shift", type=float, default=0, help=" - Amount to shift motion capture data by when aligning.")
parser.add_argument("--crop", "-c", nargs=2, metavar=("start", "stop"), help=" - Crops the corner reflectors out using start and stop positions.")
parser.add_argument("--normalize", "--norm", action="store_true", help=" - Uses the range_norm function to account for different ranges in the list of intesities")
parser.add_argument("--clim", nargs=2, help=" - Color map stuff")
parser.add_argument("--reflectors", type=int, default=1, help=" - Number of corner reflectors")
parser.add_argument("--replace_nans", "--nans", action="store_true", help = " - Whether or not to replace NaNs in motion capture data.")
parser.add_argument("--range_shift", type=float, default=0, help=" - Shifts range_bins by a certain amount.")
parser.add_argument("--use_csv", "--csv", action="store_true", help=" - Whether or not to use csv files for data.")
parser.add_argument("--motion_csv", "--motion", default=None, help=" - The motion csv to read data from. Requires use_csv to be enabled.")
parser.add_argument("--cr_csv", "--reflector", "--cr", default=None, help=" - The corner reflector csv to read data from. Requires use_csv to be enabled.")
args = parser.parse_args()

print(args)

with open(args.filename, 'rb') as f:
    data = pickle.load(f)

np.set_printoptions(threshold=np.inf)
##print(data['scan_data'][0][700:1400])
np.set_printoptions(threshold=100)

if args.use_csv:
    corner_data = pd.read_csv(args.cr_csv, header=2)
    corner_data = get_cr_pos(corner_data, 'Corner Reflector')
    motion_data = pd.read_csv(args.motion_csv, header=2)
    motion_data = get_pos_data(motion_data, 'Radar')
    data = combine_data(motion_data, data, corner_data)

print(data)

if args.replace_nans:
    data = replace_nans(data)

if args.two_dimensional_range_bins:
    data['range_bins'] = data['range_bins'][0]
    
if args.realign:
    data = range_align(data, args.range_shift)
    data = time_align(data, args.realign_thresh, args.realign_shift)
    
    plt.imshow(np.abs(data['scan_data']))
    plt.show()
    
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
        
    if args.crop is not None:
        ...
##        startx = np.zeros(int(data['scan_timestamps'][-1] - data['scan_timestamps'][0])) + float(args.crop[0])
##        starty = np.linspace(data['scan_timestamps'][0], data['scan_timestamps'][-1], int(data['scan_timestamps'][-1] - data['scan_timestamps'][0]))
##        plt.plot(startx, starty, 'r--')
##        
##        stopx = np.zeros(int(data['scan_timestamps'][-1] - data['scan_timestamps'][0])) + float(args.crop[1])
##        stopy = np.linspace(data['scan_timestamps'][0], data['scan_timestamps'][-1], int(data['scan_timestamps'][-1] - data['scan_timestamps'][0]))
##        plt.plot(stopx, stopy, 'r--')
    plt.show()

if args.crop is not None:
    data = crop(data, float(args.crop[0]), float(args.crop[1]), args.realign_thresh)

if args.normalize:
    data = range_norm(data)

bpdat = backprojection(data, args.xstart, args.xstop, args.ystart, args.ystop, args.zstart, args.zstop, args.xresolution, args.yresolution, args.zresolution, args.mode)
display_backprojected_image(bpdat, (args.xstart, args.xstop), (args.ystop, args.ystart), args.filename + ".png", "Z (m)", "X (m)", "Backprojected Image", "equal")

xlen = int((args.xstop - args.xstart) / args.xresolution) + 1
ylen = int((args.ystop - args.ystart) / args.yresolution) + 1
save_data_pickle(bpdat, np.linspace(args.xstart, args.xstop, xlen), tuple(np.linspace(args.ystart, args.ystop, ylen)), args.filename + ".bpdat")
##plt.xlabel('X (m)')
##plt.ylabel('Z (m)')
##plt.imshow(bpdat,
##    extent=(
##        args.xstart,
##        args.xstop,
##        args.ystop,
##        args.ystart))
####plt.clim(100)
##plt.colorbar()
####plt.plot(data['platform_pos'][:, 0], data['platform_pos'][:, 1], 'r-')
##plt.show()

