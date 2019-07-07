#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PulsON 440 radar data unpacking."""

__author__ = "Ramamurthy Bhagavatula, Michael Riedl"
__version__ = "1.0"
__maintainer__ = "Ramamurthy Bhagavatula"
__email__ = "ramamurthy.bhagavatula@ll.mit.edu"

# Update path
from pathlib import Path
import sys
if Path('..//').resolve().as_posix() not in sys.path:
    sys.path.insert(0, Path('..//').resolve().as_posix())

# Import required modules and methods
import argparse
from common.constants import SPEED_OF_LIGHT
from common.helper_functions import is_valid_file
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
import numpy as np
import os
import pickle
from pulson440.formats import MRM_GET_CONFIG_CONFIRM
from pulson440.constants import T_BIN, DT_0, MAX_SCAN_INFO_PACKET_SIZE, DEFAULT_CONFIG, BYTE_ORDER

class value_formatter(Formatter):
    """Tick label formatter class specific to formatting range/range bin and time/pulse number ticks 
    for displaying RTIs.
    """
    
    def __init__(self, values):
        """value_formatter class initialization method.
        
        Args:
            values (list)
                Values to reference in formatting ticks.
        """
        self.values = values
        
    def __call__(self, x, pos=0):
        """Return formatted tick label.
        
        Args:
            x (numeric)
                Tick value to format.
            
            pos (int)
                Tick position.
        
        Returns:
            tick_label (str)
                Formatted tick label.
        """
        ind = int(np.round(x))
        if ind >= len(self.values) or ind < 0:
            return ''
        return '{0:.1f}  [{1:d}]'.format(self.values[ind], ind)

def read_config_data(file_handle):
    """Read in configuration data based on platform.
    
    Args:
        file_handle (file object)
            File handle to data to unpack.
            
    Returns:
        config (dict)
            Radar configuration read from the data.
    """
    # Unpack configuration
    config = dict.fromkeys(DEFAULT_CONFIG)
    for config_field in DEFAULT_CONFIG:
        dtype = MRM_GET_CONFIG_CONFIRM['packet_def'][config_field]
        config_field_bytes = file_handle.read(dtype.itemsize)
        config[config_field] = int.from_bytes(config_field_bytes, byteorder=BYTE_ORDER, 
                signed=np.issubdtype(dtype, np.signedinteger))
    return config

def unpack(scan_data_filename):
    """Unpacks PulsOn 440 radar data from input data file.
    
    Args:
        scan_data_filename (str)
            Path and name to scan data file.
            
    Returns:
        data (dict)
            Data unpacked from data file.
    """
    with open(scan_data_filename, 'rb') as f:
        
        # Read configuration part of data
        config = read_config_data(f)

        # Compute range bins in datas
        scan_start_time = float(config['scan_start'])
        start_range = SPEED_OF_LIGHT * ((scan_start_time * 1e-12) - DT_0 * 1e-9) / 2

        # Initialize container for unpacked data
        data = dict()
        data = {'scan_data': [],
                'timestamps': [],
                'pulse_idx': None,
                'range_bins': None,
                'packet_idx': [],
                'config': config}
        single_scan_data = []
        packet_count = 0
        pulse_count = 0
        
        # Read data
        while True:
            
            # Read a single data packet and break loop if not a complete packet (in terms of size)
            packet = f.read(MAX_SCAN_INFO_PACKET_SIZE)

            if len(packet) < MAX_SCAN_INFO_PACKET_SIZE:
                break            
            
            # Get information from first packet about how scans are stored and range bins collected
            if packet_count == 0:
                num_range_bins = np.frombuffer(packet[44:48], dtype='>u4')[0]
                num_packets_per_scan = np.frombuffer(packet[50:52], dtype='>u2')[0]
                drange_bins = SPEED_OF_LIGHT * T_BIN * 1e-9 / 2
                range_bins = (start_range + drange_bins * np.arange(0, num_range_bins, 1))
            
            # Number of samples in current packet, timestamp, and packet index
            num_samples = np.frombuffer(packet[42:44], dtype='>u2')[0]
            timestamp = np.frombuffer(packet[8:12], dtype='>u4')[0]
            data['packet_idx'].append(np.frombuffer(packet[48:50], dtype='>u2')[0])
            
            # Extract radar data samples from current packet; process last packet within a scan 
            # seperately to get all data
            packet_data = np.frombuffer(packet[52:(52 + 4 * num_samples)], dtype='>i4')
            single_scan_data.append(packet_data)
            packet_count += 1
            
            if packet_count % num_packets_per_scan == 0:
                data['scan_data'].append(np.concatenate(single_scan_data))
                data['timestamps'].append(timestamp)
                single_scan_data = []
                pulse_count += 1

        # Add last partial scan if present
        if single_scan_data:
            single_scan_data = np.concatenate(single_scan_data)
            num_pad = data['scan_data'][0].size - single_scan_data.size
            single_scan_data = np.pad(single_scan_data, (0, num_pad), 'constant', constant_values=0)
            data['scan_data'].append(single_scan_data)
            data['timestamps'].append(timestamp)
            pulse_count += 1
            
        # Stack scan data into 2-D array (rows -> pulses, columns -> range bins)
        data['scan_data'] = np.stack(data['scan_data'])
        
        # Finalize entries in data
        data['timestamps'] = np.asarray(data['timestamps'])
        data['pulse_idx'] = np.arange(0, pulse_count)
        data['range_bins'] = range_bins
        data['packet_idx'] = np.asarray(data['packet_idx'])
        
        return data

def parse_args(args):
    """Input argument parser.
    
    Args:
        args (list)
            Input arguments as taken from command line execution via sys.argv[1:].
    
    Returns:
        parsed_args (namespace)
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='PulsON 440 radar data unpacking')
    parser.add_argument('scan_data_filename', help='Path and name of PulsON 440 scan data file')
    parser.add_argument('unpacked_data_filename', nargs='?', default=None, 
                        help=('Path and name of file to which data will be saved to; defaults to ' +
                              'not saving it'))
    parser.add_argument('-v', '--visualize', action='store_true', 
                        help='Plot RTI of unpacked data; will block computation')
    parsed_args = parser.parse_args(args)
    
    # Check if files are accessible
    is_valid_file(parser, parsed_args.scan_data_filename, 'r')
    is_valid_file(parser, parsed_args.unpacked_data_filename, 'w')
    
    return parsed_args

def main(args):
    """Main execution method to unpack data, visualize, and save as specified.
    
    Args:
        args (list)
            Input arguments as taken from command line execution via sys.argv[1:].
    
    Returns:
        scan_data (dict)
            Scan data unpacked from the scan data file.
    """
    # Parse input arguments
    args = parse_args(args)
    
    # Unpack log file
    scan_data = unpack(args.scan_data_filename)
    
    # Save (pickle) unpacked data if requested
    if args.unpacked_data_filename:
        with open(args.unpacked_data_filename, 'wb') as f:
            pickle.dump(scan_data, f)

    # Visualize RTI of unpacked data
    plt.ioff()
    if args.visualize:
        range_formatter = value_formatter(scan_data['range_bins'])
        pulse_formatter = value_formatter((scan_data['timestamps'] - scan_data['timestamps'][0]) / 
                                          1000)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        h_img = ax.imshow(20 * np.log10(np.abs(scan_data['scan_data'])))
        ax.set_aspect('auto')
        ax.set_title('Range-Time Intensity')
        ax.set_xlabel('Range (m) [Range Bin Number]')
        ax.set_ylabel('Time Elapsed (s) [Pulse Number]')
        ax.xaxis.set_major_formatter(range_formatter)
        ax.yaxis.set_major_formatter(pulse_formatter)
        cbar = fig.colorbar(h_img)
        cbar.ax.set_ylabel('dB')
        
        # Try to display to screen if available otherwise save to file
        try:
            plt.show()
        except:
            plt.savefig('%s.png' % os.path.splitext(args.file)[0])
            
    # Return unpacked scan_data
    return scan_data

if __name__ == "__main__":
    """Standard Python alias for command line execution."""
    main(sys.argv[1:])
    