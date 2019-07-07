#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""u-blox C94-M8P RTK GPS command and control class."""

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
import serial
import time
import datetime
from ublox_constants import SERIAL_PORT, MAX_PACKET_SIZE

# Timeout settings
READ_PACKET_TIMEOUT = 1 # Time (s) allowed between consecutive packets before raising error

# Status and control
CONTROL_FILE_NAME = "control_rtk"
STATUS_FILE_NAME = "status_rtk"

class Ublox:
    """
    Class for command and control of u-blox C94-M8P RTK GPS.
    """
    def __init__(self, serial_port=SERIAL_PORT, verbose=False):
        """
        Instance initialization.
        Inputs:
            serial_port - Name of serial port to target for connecting to 
                          device.
            verbose - Boolean flag indicating whether or not to print status
                      updates to host PC screen.
        Outputs:
            self - Instance of Ublox class.
        """
        # GPS status indicators
        self.connected = False
        self.logging = False
        
        # Connection settings
        self.connection = {
                'serial_port': serial_port, # Name of serial port
                'ser': []} # Serial port
        
        # Control and status file handles
        self.status_file = open(STATUS_FILE_NAME, "w")
        self.control_file = []
        
        # Miscellaneous
        self.verbose = verbose
    
    def update_status(self, message):
        """
        Add update to status file and print to command line if specified.
        Inputs:
            self - Instance of Ublox class.
            message - String to publish to status file and/or command line.
        Outputs:
            None
        """
        message = (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
                   " - " + message + "\n")
        self.status_file.write(message)
        self.status_file.flush()
        if self.verbose:
            print(message[:-1])
            
    def connect(self):
        """
        Connect to GPS and set up control and status files.
        Inputs:
            self - Instance of Ublox class.
        Outputs:
            None
        """
        # Try to connect to GPS
        self.update_status("Trying to connect to GPS...")
        try:
            self.connection['ser'] = serial.Serial(SERIAL_PORT)
            self.connected = True
        except:
            self.update_status("Failed to connect to GPS!")
            self.status_file.close()
            raise ConnectionError("Failed to connect to GPS!")
            
        # Set up the control file; 0 -> continue, 1 -> stop
        self.control_file = open(CONTROL_FILE_NAME, "w")
        self.control_file.write("0")
        self.control_file.close()
        self.control_file = open(CONTROL_FILE_NAME, "r+")
        self.update_status("Connected to GPS!")
        self.update_status("Control file named %s." % CONTROL_FILE_NAME)
        
    def disconnect(self):
        """
        Disconnect from GPS and close control and status files.
        Inputs:
            self - Instance of Ublox class.
        Outputs:
            None
        """
        # Try to disconnect from GPS
        self.update_status("Trying to disconnect from GPS...")
        try:
            self.connection['ser'].close()
            self.connected = False
            self.update_status("Disconnected from GPS!")
        except:
            self.update_status("Failed to disconnect from GPS!")
            raise ConnectionError("Failed to disconnect from GPS!")
        
        # Close control and status files
        self.control_file.close()
        self.status_file.close()
        
    def log(self, log_file=None, return_data=False):
        """
        Logs GPS data continuously until commanded to stop.
        Inputs:
            self - Instance of Ublox class.
            log_file - Path and name of file to save GPS data to; defaults to 
                       None such that it does not save the data to file.
            return_data - Boolean flag indicating whether or not to return
                          read data; flag exists to avoid creating large 
                          internal variables when not needed.
        Outputs:
            data - String representing the data read from the GPS; needs to 
                   unpacked to properly access scan information. Will only be
                   returned if return_data input flag is set to True.
        """
        # Check if data is being saved in some fashion
        if not return_data and not log_file:
            self.update_status("GPS data not being logged to file or " + 
                               "returned!")
            raise RuntimeError("GPS data not being logged to file or " + 
                               "returned!")
            
        # Check if GPS is connected and not already logging data
        if self.connected and not self.logging:
            self.update_status("Logging data...")
            
            # Create return data if needed
            if return_data:
                data = ''
            
            # Create save file if needed
            if log_file is not None:
                save_file = open(log_file, "wb")
            
            # Read streaming data off GPS
            start = time.time()
            while True:
                try:
                    # Try to read a packet if available
                    packet_data = self.connection['ser'].read(
                            MAX_PACKET_SIZE)
                    if return_data:
                        data += packet_data
                    if log_file is not None:
                        save_file.write(packet_data)
                    start = time.time()
                    
                    # Read until stop flag has been posted to the control file
                    stop_flag = self.control_file.read()
                    if stop_flag != "0":
                        self.control_file.seek(0)
                        self.control_file.write("0")
                        self.control_file.seek(0)
                        break
                    self.control_file.seek(0)
                    
                # Check if single packet read timeout threshold has been violated
                except:
                    if (time.time() - start) > READ_PACKET_TIMEOUT:
                        self.update_status("GPS data packet read timed out!")
                        raise RuntimeError("GPS data packet read timed out!")
                    pass
                
            self.update_status("Successfully read all the data!")
                
        else:
            self.update_status("GPS not connected or is already logging data!")
            raise ConnectionError("GPS not connected or is already logging " +
                                  "data!")
            
        # Return data if requested
        if return_data:
            return data
        