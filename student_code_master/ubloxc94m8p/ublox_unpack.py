#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
u-blox RTK log unpacking.

Usage: rtk_unpack.py [-h] [-v] rtk_log nav_data_file

Positional arguments:
  rtk_log          RTK log to unpack
  nav_data_file    File to pickle unpacked navigation data to

Optional arguments:
  -h, --help       show this help message and exit
  -v, --visualize  Visualize lon-lat-height of unpacked RTK log
"""

__author__ = "Ramamurthy Bhagavatula, Michael Riedl"
__version__ = "1.0"
__maintainer__ = "Ramamurthy Bhagavatula"
__email__ = "ramamurthy.bhagavatula@ll.mit.edu"

"""
References
[1] u-blox 8 / u-blox M8 Receiver Description Including Protocol Specification
    Document Number: UBX-13003221
    Revision: R15 (26415b7)
    Date: 6 March 2018
    https://www.u-blox.com/sites/default/files/products/documents/u-blox8-M8_ReceiverDescrProtSpec_%28UBX-13003221%29_Public.pdf
"""

# Import required modules and methods
import sys
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import is_valid_file
from math import floor
from warnings import warn
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from ublox_formats import UBX_SYNC1, UBX_SYNC2, MESSAGE_CLASS, MESSAGE_ID, \
    PAYLOAD_FORMAT, PAYLOAD_LEN

def dec_deg_to_deg_min_sec(dec_degrees, pos):
    """
    Converts coordinate degrees in decimal format to degrees-minutes-seconds 
    format for the purpose of rendering ticks using matplotlib's FuncFormatter.
    Inputs:
        dec_degrees - Coordinate degrees in decimal format.
        pos - Tick position.
    Output:
        tick_label - String to use as formatted tick label.
    """
    degrees = floor(dec_degrees)
    minutes = floor(60 * (dec_degrees - degrees))
    seconds = 3600 * (dec_degrees - degrees) - 60 * minutes
    return u"%d\N{DEGREE SIGN} %d' %.2f''" % (degrees, minutes, seconds) 
    
def checksum(buffer, checkA, checkB):
    """
    8-bit Fletcher algorithm for packet integrity checksum. Refer to [1] 
    (section 32.4 UBX Checksum, pages 135 - 136).
    Inputs:
        buffer - They byte buffer to compute the checksum over.
        checkA - The first part of the reference checksum to compare the 
                 computed value to.
        checkB - The second part of the reference checksum to compare the 
                 computed value to.         
    Outputs:
        valid - Boolean flag indicating whether or not packet checksum matches
                reference checksum.
        buffer_checkA - First part of checksum computed from input buffer.
        buffer_checkB - Second part of checksum computed from input buffer.
    """
    # Compute buffer checksum
    buffer_checkA = 0
    buffer_checkB = 0
    for byte in buffer:
        buffer_checkA = buffer_checkA + byte
        buffer_checkB = buffer_checkB + buffer_checkA
    buffer_checkA = buffer_checkA & 0xFF
    buffer_checkB = buffer_checkB & 0xFF
        
    # Compare to packet provided checksum
    valid = True
    if checkA != buffer_checkA or checkB != buffer_checkB:
        valid = False
        
    return valid, buffer_checkA, buffer_checkB
    
def read_and_check(file_handle, byte_sizes):
    """
    Return data read from input file stream in segments of specified byte sizes
    and check for complete reads.
    Inputs:
        file_handle - Handle to file to read from.
        byte_sizes - List of byte sizes to read.
    Outputs:
        complete_read - Boolean flag indicating whether or not all byte_sizes 
                        were read from input file.
        data - List of read bytes where the i-th element is the should have 
               byte_sizes[i] bytes.
    """
    # Initialize outputs
    data = []
    complete_read = True
    for byte_size in byte_sizes:
        data.append(file_handle.read(byte_size))
        if len(data[-1]) != byte_size:
            complete_read = False
            warn('Found partial packet! Hopefully caused by EOF. Saving any ' +
                 'decoded complete packets...')
            break
    return complete_read, data
    
def parse_payload(payload):
    """
    Parse payload according to its defined format.
    Inputs:
        payload - Byte buffer containing the payload to be parsed.
    Outputs:
        payload_data - Dictionary whose key-value pairs contain the parsed 
                       payload data.
    """
    # Iterate over each field in payload
    payload_data = dict.fromkeys(PAYLOAD_FORMAT.keys())
    byte_start = 0
    for field, field_format in PAYLOAD_FORMAT.items():
        field_size = field_format[0].itemsize * field_format[1]
        byte_end = byte_start + field_size
        field_bytes = payload[byte_start:byte_end]
        payload_data[field] = (field_format[2] * 
                    np.frombuffer(field_bytes, field_format[0]))
        byte_start = byte_end
        
        # Parse bitfield if required
        if field_format[3] is not None:
            binary_rep = bin(payload_data[field][0])[2:]
            binary_rep = (field_size * 8 - len(binary_rep)) * '0' + binary_rep
            bit_start = 0
            for bitfield, bitfield_format in field_format[3].items():
                bitfield_size = bitfield_format[0]
                bit_end = bit_start + bitfield_size
                bitfield_bits = binary_rep[bit_start:bit_end]
                payload_data[bitfield] = int(bitfield_bits, 2)
                bit_start = bit_end
        
    return payload_data
    
def unpack_rtk(file):
    """
    Unpack RTK log.
    Inputs:
        file - String containing path and name to file to unpack.
    Outptus:
        nav_data - Dictionary whose key-value pairs contain the unpacked 
                   navigation data from file.
    """
    with open(file, 'rb') as f:
        
        # Initialize output navigation data and NAV-PVT solution count
        nav_data = None
        nav_pvt_count = 0
        
        # Iterate through log
        while True:
            #print(nav_pvt_count)
            #nav_pvt_count += 1
            
            # Try to read one byte of data assumed to be first part of next 
            # packet's header; unpack is done if EOF encountered at this point
            ubx_sync1 = f.read(1)
            if ubx_sync1 == b'':
                break
            
            # Check for first UBX sync character; assumes all packets are UBX 
            # and inefficiently parses other protocols 1 byte at a time
            # TODO: Address mixed protocol logs, i.e., UBX and NMEA messages so
            #       as to improve the unpacking speed for such logs
            if ubx_sync1 == UBX_SYNC1:
                
                # Check for partial packet by checking second UBX sync character
                complete_read, data = read_and_check(f, [1])
                if not complete_read:
                    break
                ubx_sync2 = data[0]
                
                # Check second UBX sync character
                if ubx_sync2 == UBX_SYNC2:
                    
                    # Check for partial packet by reading in remainder of UBX header
                    complete_read, data = read_and_check(f, [1, 1, 2])
                    if not complete_read:
                        break
                    message_class = data[0]
                    message_id = data[1]
                    payload_len = np.frombuffer(data[2], dtype='u2')[0]
                    
                    # Check for desired message class and ID
                    if (message_class == MESSAGE_CLASS and message_id == MESSAGE_ID):
                        
                        # Check for packet error by comparing decoded payload
                        # length to expected one
                        if payload_len == PAYLOAD_LEN:
                            
                            # Check for partial packets by reading in 
                            # payload and checksum
                            complete_read, data = read_and_check(f, [payload_len, 1, 1])
                            if not complete_read:
                                break
                            payload = data[0]
                            packet_checkA = np.frombuffer(data[1], dtype='u1')[0]
                            packet_checkB = np.frombuffer(data[2], dtype='u1')[0]
                            
                            # Parse payload
                            payload_data = parse_payload(payload)
                            
                            # Compute UTC time
                            # TODO: Section 8.6 UTC Representation in [1] 
                            #       implies that handling of nano requires more
                            #       than how it is handled here
                            sec = min(max([payload_data['sec'] + payload_data['nano'] * 1e-9, 0]), 60)
                            utcTime = '%d-%02d-%02dT%02d:%02d:%06.3f' % (
                                    payload_data['year'], payload_data['month'],
                                    payload_data['day'], payload_data['hour'], 
                                    payload_data['min'], sec)
                            utcTime = np.datetime64(utcTime)
                            
                            # Validate checksum
                            (checksum_pass, payload_checkA, payload_checkB) = checksum(
                                     message_class + message_id + payload_len + payload, 
                                     packet_checkA, packet_checkB)
                            
                            # Update output navigation data; initialize if needed
                            if nav_data is None:
                                nav_data = {field: [] for field in payload_data.keys()}
                                nav_data['packetCheckA'] = []
                                nav_data['packetCheckB'] = []
                                nav_data['payloadCheckA'] = []
                                nav_data['payloadCheckB'] = []
                                nav_data['checksumPass'] = []
                                nav_data['quality'] = []
                                nav_data['utcTime'] = []
                            for field, value in payload_data.items():
                                nav_data[field].append(value)
                            nav_data['packetCheckA'].append(packet_checkA)
                            nav_data['packetCheckB'].append(packet_checkB)
                            nav_data['payloadCheckA'].append(payload_checkA)
                            nav_data['payloadCheckB'].append(payload_checkB)
                            nav_data['checksumPass'].append(checksum_pass)
                            nav_data['utcTime'].append(utcTime)
                            
                            # Update NAV-PVT solution count
                            nav_pvt_count += 1
                            
                        else:
                            warn('Found packet error! Decoded packet length ' +
                                 'does not match message defintion. Saving ' + 
                                 'any decoded complete packets...')
                            break
                        
                    # Skip other UBX messages
                    else:
                        f.seek(payload_len + 2, 1)
                        
    # Finalize output navigation data
    for field, value in nav_data.items():
        nav_data[field] = np.squeeze(np.reshape(value, (nav_pvt_count, -1)))
    nav_data['quality'] = np.zeros_like(nav_data['carrSoln'])
    nav_data['quality'][nav_data['carrSoln'] == 1] = 2
    nav_data['quality'][nav_data['carrSoln'] == 2] = 1
    
    return nav_data

def parse_args(args):
    """
    Input argument parser.
    Inputs:
        args - List of input arguments as taken from command line execution via
               sys.argv[1:].
    Outputs:
        parsed_args - Namespace populated with attributes taken from argparse
                      configuration and input arguments.
    """
    parser = argparse.ArgumentParser(description='U-blox RTK log unpacking')
    parser.add_argument('rtk_log_file', type=str, help='RTK log to unpack')
    parser.add_argument('nav_data_file', type=str, 
                        help=('File to save (pickle) unpacked navigation ' +
                              'data to; defaults to not saving it'))
    parser.add_argument('-v', '--visualize', action='store_true', 
                        help='Visualize lon-lat-height of unpacked RTK log')
    
    # Parse input arguments
    parsed_args = parser.parse_args(args)
    
    # Check the file inputs
    is_valid_file(parser, parsed_args.rtk_log_file, 'r')
    is_valid_file(parser, parsed_args.nav_data_file, 'w')
    
    return parsed_args

def main(args):
    """
    Main execution method to unpack data, visualize, and save as specified.
    Inputs:
        args - List of input arguments as taken from command line execution via
               sys.argv[1:].
    Outputs:
        nav_data - Dictionary whose key-value pairs contain the data unpacked 
                   from the data file.
    """
    # Parse input arguments
    parsed_args = parse_args(args)
    
    # Unpack RTK log
    nav_data = unpack_rtk(parsed_args.rtk_log_file)
    
    # Pickle unpacked data
    with open(parsed_args.nav_data_file, 'wb') as f:
        pickle.dump(nav_data, f)
        
    # Visualize unpacked lat-lon-height RTK
    if parsed_args.visualize:
        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(nav_data['lon'], nav_data['lat'], nav_data['height'] / 1e3, 
                marker='o', label='Logged Path')
        ax.plot([nav_data['lon'][0]], [nav_data['lat'][0]], 
                [nav_data['height'][0] / 1e3], marker='s', label='Start')
        ax.plot([nav_data['lon'][-1]], [nav_data['lat'][-1]], 
                [nav_data['height'][-1] / 1e3], marker='s', label='Stop')
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_zlabel('Height (m)')
        ax.legend()
        formatter = FuncFormatter(dec_deg_to_deg_min_sec)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        # Try to display to screen if available otherwise save to file
        try:
            plt.show()
        except:
            plt.savefig('%s.png' % os.path.splitext(args.nav_data_file)[0])
        
    # Return unpacked data
    return nav_data

if __name__ == "__main__":
    """
    Standard Python alias for command line execution.
    """
    main(sys.argv[1:])
    