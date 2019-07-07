#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
u-blox RTK GPS data logging using a Raspberry Pi.

"""

__author__ = "Ramamurthy Bhagavatula, Michael Riedl"
__version__ = "1.0"
__maintainer__ = "Ramamurthy Bhagavatula"
__email__ = "ramamurthy.bhagavatula@ll.mit.edu"

# Import required modules and methods
import sys
import argparse
from ublox import Ublox
from helper_functions import is_valid_file

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
    parser = argparse.ArgumentParser(
            description=('u-blox RTK GPS data logging using a Raspberry Pi. ' +
                         'After starting logging, users can stop logging by ' +
                         'posting any non-zero value to the control file ' + 
                         'created upon successful connection to the GPS. On ' +
                         'Unix systems it is recommended that users run ' + 
                         'this script as a background process by appending ' + 
                         '\" &\" to the end of the command line call so as ' + 
                         'to allow usage of the control file.'))
    parser.add_argument('log_file', type=str, nargs='?', default=None, 
                        help=('Name of file to save RTK log to; defaults to ' +
                              'untitled_gps_log.rtk'))
    parser.add_argument('-r', '--return_data', action='store_true',
                        help=('Return collected data; only useful if ' +
                              'calling this method in other code'))
    
    # Parse input arguments
    parsed_args = parser.parse_args(args)
    
    # Check the file input
    is_valid_file(parser, parsed_args.log_file, 'w')
    
    return parsed_args

def main(args):
    """
    Main execution method to log u-blox RTK GPS data.
    Inputs:
        args - List of input arguments as taken from command line execution via
               sys.argv[1:].
    Outputs:
        nav_data - Dictionary whose key-value pairs contain the data unpacked 
                   from the data file.
    """
    # Parse input arguments
    parsed_args = parse_args(args)
    
    # Create PulsON440 object
    gps = Ublox()
    
    # Connect to the GPS
    gps.connect()
    
    # Log GPS data
    try:
        if parsed_args.return_data:
            data = gps.log(parsed_args.log_file, True)
        else:
            gps.log(parsed_args.log_file, False)
            
    except Exception as e:
        gps.disconnect()
        raise e
        
    # Disconnect GPS
    gps.disconnect()
    
    # Return data if requested
    if parsed_args.return_data:
        return data
    
if __name__ == "__main__":
    """Standard Python alias for command line execution."""
    main(sys.argv[1:])
    