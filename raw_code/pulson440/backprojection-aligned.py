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
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"][0] if twodimbins else data["range_bins"]
    # The number of pixels per dimension
    alen = int((stop - start) / resolution) + 1
    xlen = alen
    ylen = alen
    # Create the return data array with datatype complex128
    return_data = np.zeros((xlen, ylen), dtype=np.complex128)
    # Loop over each scan and add the appropriate intensities to each pixel through np.interp
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number] # Take the position of the radar at the time of the current scan
        meshgrid = np.asarray(np.meshgrid(np.linspace(start, stop, xlen), np.linspace(start, stop, ylen))) # Create a 2D grid
        points = np.concatenate((meshgrid, np.zeros((1, xlen, ylen)))).transpose(1, 2, 0) # Add a Z-dimesion and fill it with zeros
        distances = np.linalg.norm(points - pos, axis=2) # Take the distance of each coordinate from the radar
        interp = np.interp(distances, range_bins, scan_data[scan_number]).reshape(xlen, ylen) # Interpolate on those distances using range_bin data and scan_data
        return_data += np.flipud(interp) # Flip the returned intensities and add the intensities to the return_data array
    return np.abs(return_data)

def backprojection3dfast(data, start=-3, stop=3, resolution=0.05, twodimbins=False):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"][0] if twodimbins else data["range_bins"]
    alen = int((stop - start) / resolution) + 1
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
##    print(range_bins)
    alen = int((stop - start) / resolution) + 1
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

def MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
##    print(one_way_range == first_value)
    indices = (one_way_range != first_value).nonzero()
    # print(indices)
    tkf_scan_num = indices[0][0]
    return tkf_scan_num + 1


# function within a function; finds the time stamp at which the drone takes off relative to the radar's timer
def RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
    for k in range(1, num_scans):
        current = scan_data[k, cr_first_rbin]
        previous = scan_data[k-1, cr_first_rbin]
        if np.abs(current - previous) > 0.5:
            return k

def RD_tkf_indexn(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
##    np.set_printoptions(threshold=np.inf)
##    print(np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]))
##    np.set_printoptions(threshold=1000)
    return (np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > 5).nonzero()[0][0]

def motion_align(data):
    scan_data = data['scan_data']
    platform_pos = data['platform_pos']
    range_bins = data['range_bins']
    corner_reflector_pos = data['corner_reflector_pos']
    motion_timestamps = data['motion_timestamps']
    scan_timestamps = data['scan_timestamps'] - data['scan_timestamps'][0]##    print(MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos), RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos), RD_tkf_indexn(scan_data, platform_pos, range_bins, corner_reflector_pos))

    RD_change_time = scan_timestamps[RD_tkf_indexn(scan_data, platform_pos, range_bins, corner_reflector_pos)]
    MC_change_time = motion_timestamps[MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos)]

    print(RD_change_time, MC_change_time)

    newdata = data.copy()

    newdata['scan_timestamps'] -= newdata['scan_timestamps'][0]
    newdata['motion_timestamps'] += RD_change_time - MC_change_time

##    print(newdata)

    pos_x = newdata['platform_pos'][:, 0]
    pos_y = newdata['platform_pos'][:, 1]
    pos_z = newdata['platform_pos'][:, 2]

    realigned_pos_x = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_x)
    realigned_pos_y = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_y)
    realigned_pos_z = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_z)

    newdata['platform_pos'] = np.column_stack((realigned_pos_x, realigned_pos_y, realigned_pos_z))
    newdata['motion_timestamps'] = newdata['scan_timestamps']
    print(newdata['scan_timestamps'])
    return newdata

def range_norm(scan_data, range_bins):
    norm_scan_data = np.copy(scan_data)
    norm_scan_data *= ((range_bins / range_bins[0]) ** 4)
    return norm_scan_data

parser = argparse.ArgumentParser(description=" - Runs backprojection on SAR data.")
parser.add_argument("--filename", "-f", type=str, required=True, help=" - The SAR data filename")
parser.add_argument("--start", type=float, default=-3, help=" - The start of the coordinate plane")
parser.add_argument("--stop", type=float, default=3, help=" - The end of the coordinate plane")
parser.add_argument("--resolution", "--res", type=float, default=0.05, help=" - The resolution of the SAR image")
parser.add_argument("--two_dimensional_range_bins", "--2D-range-bins",  "--2D_bins", action="store_true", help=" - Enables use of weird 2D range_bins.")
parser.add_argument("--mode", "-m", type=str, default="2dfast", help=" - The mode to run back projection in.")
parser.add_argument("--realign", "--align", action="store_true", help=" - Realigns the motion capture data and the radar data")
args = parser.parse_args()

with open(args.filename, 'rb') as f:
    data = pickle.load(f)


data = motion_align(data)

range_norm(data['scan_data'], data['range_bins'])

print(args)
##print(data)
if args.realign:
    bpdat = backprojection(motion_align(data), args.start,args.stop, args.resolution, args.two_dimensional_range_bins, args.mode)
else:
    bpdat = backprojection(data, args.start,args.stop, args.resolution, args.two_dimensional_range_bins, args.mode)
plt.imshow(bpdat)
plt.show()


plt.imshow(np.abs(data['scan_data']),
           extent=(
                   data['range_bins'][0, 0],
                   data['range_bins'][0, -1],
                   data['scan_timestamps'][-1] - data['scan_timestamps'][0],
                   0))

plt.xlabel('Range (m)')
plt.ylabel('Elapsed Time (s)')

print(len(data['motion_timestamps']) == len(data['platform_pos']))

##ranges = get_ranges(data)
r1 = np.sqrt(np.sum(
        (data['platform_pos'] - data['corner_reflector_pos'][0, :])**2, 1))
plt.plot(r1, data['motion_timestamps'], 'r--', label='Corner Reflector 1')
plt.show()
=======
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
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"][0] if twodimbins else data["range_bins"]
    # The number of pixels per dimension
    alen = int((stop - start) / resolution) + 1
    xlen = alen
    ylen = alen
    # Create the return data array with datatype complex128 
    return_data = np.zeros((xlen, ylen), dtype=np.complex128)
    # Loop over each scan and add the appropriate intensities to each pixel through np.interp
    for scan_number in range(len(platform_pos)):
        pos = platform_pos[scan_number] # Take the position of the radar at the time of the current scan
        meshgrid = np.asarray(np.meshgrid(np.linspace(start, stop, xlen), np.linspace(start, stop, ylen))) # Create a 2D grid
        points = np.concatenate((meshgrid, np.zeros((1, xlen, ylen)))).transpose(1, 2, 0) # Add a Z-dimesion and fill it with zeros
        distances = np.linalg.norm(points - pos, axis=2) # Take the distance of each coordinate from the radar
        interp = np.interp(distances, range_bins, scan_data[scan_number]).reshape(xlen, ylen) # Interpolate on those distances using range_bin data and scan_data
        return_data += np.flipud(interp) # Flip the returned intensities and add the intensities to the return_data array
    return np.abs(return_data)

def backprojection3dfast(data, start=-3, stop=3, resolution=0.05, twodimbins=False):
    scan_data = data["scan_data"]
    platform_pos = data["platform_pos"]
    range_bins = data["range_bins"][0] if twodimbins else data["range_bins"]
    alen = int((stop - start) / resolution) + 1
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
##    print(range_bins)
    alen = int((stop - start) / resolution) + 1
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

def MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
##    print(one_way_range == first_value)
    indices = (one_way_range != first_value).nonzero()
    # print(indices)
    tkf_scan_num = indices[0][0]
    return tkf_scan_num + 1


# function within a function; finds the time stamp at which the drone takes off relative to the radar's timer
def RD_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
    for k in range(1, num_scans):
        current = scan_data[k, cr_first_rbin]
        previous = scan_data[k-1, cr_first_rbin]
        if np.abs(current - previous) > 0.5:
            return k

def RD_tkf_indexn(scan_data, platform_pos, range_bins, corner_reflector_pos):
    one_way_range = np.sqrt(np.sum(np.square(platform_pos - corner_reflector_pos[0]), axis=1))
    first_value = one_way_range[0]
    cr_first_rbin = np.argmin(np.abs(first_value - range_bins))
    num_scans = len(scan_data)
    return (np.abs(scan_data[:, cr_first_rbin] - scan_data[0, cr_first_rbin]) > 5).nonzero()[0][0]

def motion_align(data):
    scan_data = data['scan_data']
    platform_pos = data['platform_pos']
    range_bins = data['range_bins']
    corner_reflector_pos = data['corner_reflector_pos']
    motion_timestamps = data['motion_timestamps']
    scan_timestamps = data['scan_timestamps'] - data['scan_timestamps'][0]
    
    RD_change_time = scan_timestamps[RD_tkf_indexn(scan_data, platform_pos, range_bins, corner_reflector_pos)]
    MC_change_time = motion_timestamps[MC_tkf_index(scan_data, platform_pos, range_bins, corner_reflector_pos)]
    
    print(RD_change_time, MC_change_time)
    
    newdata = data.copy()
    
    newdata['scan_timestamps'] -= newdata['scan_timestamps'][0]
    newdata['motion_timestamps'] += RD_change_time - MC_change_time
        
    pos_x = newdata['platform_pos'][:, 0]
    pos_y = newdata['platform_pos'][:, 1]
    pos_z = newdata['platform_pos'][:, 2]

    realigned_pos_x = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_x)
    realigned_pos_y = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_y)
    realigned_pos_z = np.interp(newdata['scan_timestamps'], newdata['motion_timestamps'], pos_z)
    
    newdata['platform_pos'] = np.column_stack((realigned_pos_x, realigned_pos_y, realigned_pos_z))
    newdata['motion_timestamps'] = newdata['scan_timestamps']
    return newdata

def range_norm(scan_data, range_bins):
    norm_scan_data = np.copy(scan_data)
    norm_scan_data *= ((range_bins / range_bins[0]) ** 4)
    return norm_scan_data

parser = argparse.ArgumentParser(description=" - Runs backprojection on SAR data.")
parser.add_argument("--filename", "-f", type=str, required=True, help=" - The SAR data filename")
parser.add_argument("--start", type=float, default=-3, help=" - The start of the coordinate plane")
parser.add_argument("--stop", type=float, default=3, help=" - The end of the coordinate plane")
parser.add_argument("--resolution", "--res", type=float, default=0.05, help=" - The resolution of the SAR image")
parser.add_argument("--two_dimensional_range_bins", "--2D-range-bins",  "--2D_bins", action="store_true", help=" - Enables use of weird 2D range_bins.")
parser.add_argument("--mode", "-m", type=str, default="2dfast", help=" - The mode to run back projection in.")
parser.add_argument("--realign", "--align", action="store_true", help=" - Realigns the motion capture data and the radar data")
args = parser.parse_args()

with open(args.filename, 'rb') as f:
    data = pickle.load(f)

print(args)
##print(data)
if args.realign:
    bpdat = backprojection(motion_align(data), args.start,args.stop, args.resolution, args.two_dimensional_range_bins, args.mode)
else:
    bpdat = backprojection(data, args.start,args.stop, args.resolution, args.two_dimensional_range_bins, args.mode)
plt.imshow(bpdat)
plt.show()

data = motion_align(data)

plt.imshow(np.abs(data['scan_data']),
           extent=(
                   data['range_bins'][0, 0],
                   data['range_bins'][0, -1],
                   data['scan_timestamps'][-1] - data['scan_timestamps'][0],
                   0))

plt.xlabel('Range (m)')
plt.ylabel('Elapsed Time (s)')

print(len(data['motion_timestamps']) == len(data['platform_pos']))

r1 = np.sqrt(np.sum(
        (data['platform_pos'] - data['corner_reflector_pos'][0, :])**2, 1))
r2 = np.sqrt(np.sum(
        (data['platform_pos'] - data['corner_reflector_pos'][1, :])**2, 1))
plt.plot(r1, data['motion_timestamps'], 'r--', label='Corner Reflector 1')
plt.plot(r2, data['motion_timestamps'], 'r--', label='Corner Reflector 2')
plt.show()
