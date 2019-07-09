#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PulsON 440 radar command and control class."""

__author__ = 'Ramamurthy Bhagavatula, Michael Riedl'
__version__ = '1.0'
__maintainer__ = 'Ramamurthy Bhagavatula'
__email__ = 'ramamurthy.bhagavatula@ll.mit.edu'

"""References
[1] Monostatic Radar Application Programming Interface (API) Specification
    PulsON (R) 400 Series
    Version: 1.2.2
    Date: January 2015
    https://timedomain.com/wp-content/uploads/2015/12/320-0298E-MRM-API-Specification.pdf
"""
# Update path
from pathlib import Path
import sys
if Path('..//').resolve().as_posix() not in sys.path:
    sys.path.insert(0, Path('..//').resolve().as_posix())
    
# Import required modules and methods
from common.constants import SPEED_OF_LIGHT
from collections import OrderedDict
from copy import deepcopy
import math
import logging
import numpy as np
from pulson440.constants import BYTE_ORDER, DEFAULT_SETTINGS, DEFAULT_CONFIG, MAX_PACKET_SIZE, \
    FOREVER_SCAN_COUNT, STOP_SCAN_COUNT, MIN_SCAN_COUNT, CONTINUOUS_SCAN_INTERVAL, DT_MIN, \
    T_BIN, DN_BIN, SEG_NUM_BINS, UDP_IP_HOST, UDP_IP_RADAR, UDP_PORT_RADAR, REC_SCAN_RES, \
    REC_PERSIST_FLAG
from pulson440.formats import MRM_CONTROL_CONFIRM, MRM_CONTROL_REQUEST, MRM_GET_CONFIG_CONFIRM, \
    MRM_GET_CONFIG_REQUEST, MRM_SET_CONFIG_CONFIRM, MRM_SET_CONFIG_REQUEST
import socket
import time
import yaml

# Control file
CONTROL_FILENAME = 'control_radar'

class PulsON440:
    """Class for command and control of PulsON 440 radar."""
    
    def __init__(self, logger=None, udp_ip_host=UDP_IP_HOST, udp_ip_radar=UDP_IP_RADAR, 
                 udp_port_radar=UDP_PORT_RADAR):
        """Instance initialization.
        
        Args:
            logger (logging.Logger)
                Configured logger.
                
            udp_ip_host (str)
                IP address of the host computer, i.e., the machine that commands the radar. Defaults
                to pulson440_constants.UDP_IP_HOST.
                
            udp_ip_radar (str)
                String defining the IP address of the radar. Defaults to 
                pulson440_constants.UDP_IP_RADAR.
                
            udp_port_radar (int)
                Port on radar that the host computer should target when creating the UDP socket.
                Defaults to pulson440_constants.UDP_PORT_RADAR.
        """
        # Radar status indicators
        self.connected = False
        self.collecting = False
        
        # Radar system parameters
        self.N_bin = [] # Number of bins in scan
        
         # Connection s`ettings

        self.connection = {
                'udp_ip_host': udp_ip_host, # Host (computer) IP address
                'udp_ip_radar': udp_ip_radar, # Radar IP address
                'udp_port_radar': udp_port_radar, # Radar port
                'sock': []} # UDP socket
        
        # User radar settings; partially higher abstraction than the radar's internal configuration;
        self.settings = {key: value['default'] for key, value in DEFAULT_SETTINGS.items()}
        
        # Radar internal configuration
        self.config = DEFAULT_CONFIG
        
        # Logger
        self._logger = None
        self.logger = logger
        
        # Message counter
        self.message_count = 0
        
        # Control file
        self.control_file_handle = []
        
    def __del__(self):
        """Clean up actions upon object deletion."""
        self.disconnect()
        
    """logger property decorators. Setter validates logger's types is a valid logger type."""
    @property
    def logger(self):
        return self._logger
    
    @logger.setter
    def logger(self, value):
        if value is None:
            self._logger = logging.getLogger('trash')
            self._logger.propagate = False
        elif not issubclass(type(value), logging.getLoggerClass()):
            raise TypeError('Specified logger of incorrect type; expecting subclass of ' + 
                            'logging.Logger!')
        else:
            self._logger = value
    
    def read_settings_file(self, settings_file='radar_settings.yml'):
        """Read user specified radar settings file.
        
        Args:
            settings_file (str)
                Path and name of radar settings file.
                
        Raises:
            ValueError if setting is out of bounds.
        """
        self.logger.info('Reading settings from \'{0}\'...'.format(settings_file))
        with open(settings_file, 'r') as f:
            radar_settings = yaml.load(f)
        self.logger.info('Read following radar settings --> {0}'.format(radar_settings))
            
        # Iterate over each user setting and check bounds if applicable
        for setting, value in radar_settings.items():
            if setting in DEFAULT_SETTINGS:
                if ('bounds' in DEFAULT_SETTINGS[setting] and 
                    DEFAULT_SETTINGS[setting]['bounds'] is not None):
                    bounds = DEFAULT_SETTINGS[setting]['bounds']
                    if not (bounds[0] <= value <= bounds[1]):
                        raise ValueError('Radar setting \'{0}\' is out of bounds!'.format(setting))
                self.settings[setting] = value
            else:
                self.settings[setting] = value
                
        # Update radar configuration
        self.logger.debug('Following radar settings being used --> {0}'.format(self.settings))
        self.settings_to_config()
    
    def settings_to_config(self):
        """Translate radar settings into radar configuration."""
        # Based on the specified start and stop ranges determine the scan start and stop times
        scan_start = (2 * float(self.settings['range_start']) / (SPEED_OF_LIGHT / 1e9) + 
                      self.settings['dT_0'])
        scan_stop = (2 * float(self.settings['range_stop']) / (SPEED_OF_LIGHT / 1e9) + 
                     self.settings['dT_0'])
        N_bin = (scan_stop - scan_start) / T_BIN
        N_bin = DN_BIN * math.ceil(N_bin / DN_BIN)
        scan_start = math.floor(1000 * DT_MIN * math.floor(scan_start / DT_MIN))
        scan_stop = N_bin * T_BIN + scan_start / 1000
        scan_stop = math.floor(1000 * DT_MIN * math.ceil(scan_stop / DT_MIN))
        
        # Update radar configuration
        self.N_bin = N_bin
        self.config['scan_start'] = scan_start
        self.config['scan_stop'] = scan_stop
        self.config['pii'] = self.settings['pii']
        self.config['tx_gain_ind'] = self.settings['tx_gain_ind']
        self.config['code_channel'] = self.settings['code_channel']
        self.config['node_id'] = self.settings['node_id']
        self.config['persist_flag'] = self.settings['persist_flag']
        self.logger.debug(
                'Settings parsed into following configuration --> {0}'.format(self.config))
        
    def config_to_bytes(self):
        """Converts radar configuration to bytes so it can be written to file.

        Returns:
            config_bytes (bytes)
                The current radar configuration (as stored in instance) represented as bytes.
        """
        # Add all configuration fields
        config_bytes = b''
        for config_field, config_value in self.config.items():
            dtype = MRM_GET_CONFIG_CONFIRM['packet_def'][config_field]
            config_bytes += (config_value).to_bytes(length=dtype.itemsize, byteorder=BYTE_ORDER,
                    signed=np.issubdtype(dtype, np.signedinteger))
        return config_bytes

    def encode_host_to_radar_message(self, raw_payload, message_format):
        """Encode host to radar message.
        
        Args:
            raw_payload (dict)
                Specifies the payload to encode. Each key must match exactly a key in packet_def
                contained in message_format.
            
            message_format (dict)
                Message format as defined in formats.py. Primary keys are message_type and 
                packet_def.
        
        Returns:
            message (bytes)
                The payload encoded into a byte sequence for transmission to the radar.
                
        Raises:
            KeyError if payload does not contain key that must be user defined.
        """
        # Make a deep copy of payload to avoid malforming original
        payload = deepcopy(raw_payload)

        # Update payload w/ message type and ID
        payload['message_type'] = message_format['message_type']
        payload['message_id'] = self.message_count
        
        # Add all packet fields to message
        message = b''
        for packet_field in message_format['packet_def'].keys():
            dtype = message_format['packet_def'][packet_field][0]
            default_value = message_format['packet_def'][packet_field][1]
            
            # Check if current packet field is in payload
            if packet_field not in payload:
                if default_value is None:
                    raise KeyError('Payload for message type {0} missing field {1}'.format(
                            message_format['message_type'], packet_field))
                else:
                    payload[packet_field] = default_value
                    
            # Add current packet field's payload value onto message
            message += (payload[packet_field]).to_bytes(length=dtype.itemsize, byteorder=BYTE_ORDER, 
                       signed=np.issubdtype(dtype, np.signedinteger))
        return message
        
    @staticmethod
    def decode_radar_to_host_message(message, message_format):
        """Decode radar to host message.
        
        Args:
            message (bytes)
                Message byte sequence received from radar.
                
            message_format (dict)
                Message format as defined in formats.py. Primary keys are message_type and 
                packet_def.
                
        Returns:
            payload (dict)
                Payload decoded from message received from radar.
        """
        # Initialize decoded payload
        payload = OrderedDict.fromkeys(message_format['packet_def'])
        
        # Iterate over each field in packet definition
        byte_counter = 0
        for packet_field, dtype in message_format['packet_def'].items():
            num_bytes = dtype.itemsize
            payload[packet_field] = int.from_bytes(message[byte_counter:(byte_counter + num_bytes)],
                   byteorder=BYTE_ORDER, signed=np.issubdtype(dtype, np.signedinteger))
            byte_counter += num_bytes
        return payload
        
    def connect(self):
        """Connect to radar and set up control file.
        
        Raises:
            RuntimeError if fails to connect to radar.
        """
        # Try to connect to radar
        self.logger.info('Trying to connect to radar...')
        try:
            self.connection['sock'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.connection['sock'].setblocking(False)
            self.connection['sock'].bind((self.connection['udp_ip_host'], 
                            self.connection['udp_port_radar']))
            self.connected = True
        except:
            raise RuntimeError('Failed to connect to radar!')
            
        # Set up the control file; 0 -> continue, 1 -> stop
        self.control_file_handle = open(CONTROL_FILENAME, 'w')
        self.control_file_handle.write('0')
        self.control_file_handle.close()
        self.control_file_handle = open(CONTROL_FILENAME, 'r+')
        self.logger.info('Connected to radar!')
    
    def disconnect(self):
        """Disconnect from radar and close control file.
        
        Raises:
            RuntimeError if fails to disconnect from radar.
        """
        # Try to disconnect from radar if needed
        if not self.connected:
            self.logger.info('Cnnnot disconnect, no radar connected!')
        else:
            self.logger.info('Trying to disconnect from radar...')
            try:
                if self.collecting:
                    self.scan_request(scan_count=STOP_SCAN_COUNT)
                self.connection['sock'].close()
                self.connected = False
                self.logger.info('Disconnected from radar!')
            except:
                raise RuntimeError('Failed to disconnect from radar!')
        
        # Close control file
        if self.control_file_handle:
            self.control_file_handle.close()
    
    def get_radar_config(self):
        """Get configuration from radar.
        
        Returns:
            status_flag (int)
                Status flag indicating success/failure of get configuration request. Any non-zero 
                value is a failure.
                
        Raises:
            RuntimeError if radar not already connected.
            RuntimeError if fails to receive radar configuration within timeout.
        """
        self.logger.info('Requesting radar configuration...')
        # Make sure radar is connected
        if self.connected:
            
            # Request the current radar configuration
            payload = {}
            message = self.encode_host_to_radar_message(payload, MRM_GET_CONFIG_REQUEST)
            self.connection['sock'].sendto(message, 
                           (self.connection['udp_ip_radar'], self.connection['udp_port_radar']))
            
            # Wait for radar configuration within the timeout
            start = time.time()
            status_flag = -1
            while (time.time() - start) < self.settings['get_config_timeout']:
                try:
                    message, addr = self.connection['sock'].recvfrom(MAX_PACKET_SIZE)
                    payload = self.decode_radar_to_host_message(message, MRM_GET_CONFIG_CONFIRM)
                    self.config = OrderedDict([(key, payload[key]) for key in DEFAULT_CONFIG])
                    status_flag = payload['status']
                    break
                except:
                    pass
            if status_flag == -1:
                raise RuntimeError('Get radar configuration timed out!')
            elif status_flag != 0:
                raise RuntimeError(('Failed to get radar configuration with error code ' + 
                                    '{0}!').format(status_flag))
            self.logger.info('Get radar configuration successful!')
            self.logger.debug('Radar configuration received --> {0}'.format(self.config))
            return status_flag
        
        else:
            raise RuntimeError('Radar not connected!')
        
    def set_radar_config(self):
        """Set radar configuration based on user settings.
        
        Returns:
           status_flag (int)
                Status flag indicating success/failure of set configuration request. Any non-zero 
                value is a failure.
        
        Raises:
            RuntimeError if radar not already connected.
            RuntimeError if fails to send radar configuration within timeout.
        """
        # Make sure radar is connected
        self.logger.info('Setting radar configuration...')
        if self.connected:
            
            # Determine desired configuration from user settings
            self.settings_to_config()
            
            # Scan resolution; API states that any value aside from 32 will likely cause undesired 
            # behavior so overwrite it
            if self.config['scan_res'] != REC_SCAN_RES:
                self.logger.warning('Overriding specified scan resolution of {0} with ' + 
                                    'recommended value of {1}'.format(self.config['scan_res'], 
                                                          REC_SCAN_RES))
                self.config['scan_res'] = REC_SCAN_RES
            
            # Configuration persistence flag
            if self.config['persist_flag'] != REC_PERSIST_FLAG:
                self.logger.warning('Specified persist flag value of {0} not the recommended '
                                    'value of {1}'.format(self.config['persist_flag'], 
                                              REC_PERSIST_FLAG))
            
            # Encode configuration into message and send
            message = self.encode_host_to_radar_message(self.config, MRM_SET_CONFIG_REQUEST)
            self.connection['sock'].sendto(message, 
                           (self.connection['udp_ip_radar'], self.connection['udp_port_radar']))
            
            # Poll for configuration set confirmation from radar within timeout 
            start = time.time()
            status_flag = -1
            while (time.time() - start) < self.settings['set_config_timeout']:
                try:
                    message, addr = self.connection['sock'].recvfrom(MAX_PACKET_SIZE)
                    payload = self.decode_radar_to_host_message(message, MRM_SET_CONFIG_CONFIRM)
                    status_flag = payload['status']
                    break
                except:
                    pass
            if status_flag == -1:
                raise RuntimeError('Set radar configuration timed out!')
            elif status_flag != 0:
                raise RuntimeError(('Failed to set radar configuration with error code ' + 
                                    '{0}!').format(status_flag))
            self.logger.info('Set radar configuration successful!')
            self.logger.debug('Radar configuration set --> {0}'.format(self.config))
            return status_flag
        
        else:
            raise RuntimeError('Radar not connected!')
            
    def scan_request(self, scan_count, scan_interval=CONTINUOUS_SCAN_INTERVAL):
        """Initiate a set of scans by the radar.
        
        Args:
            scan_count (int)
                Number of scans to request; refer to [1] for details.
            
            scan_interval (int)
                Interval between sequential scans (us); defaults to CONTINUOUS_SCAN_INTERVAL for 
                continuous scanning.
        
        Returns:
            status_flag (int)
                Status flag indicating success/failure of get configuration request. Any non-zero 
                value is a failure.
        
        Raises:
            RuntimeError if radar not already connected.
            ValueError if scan_count is not between MIN_SCAN and CONTINUOUS_SCAN.
            RuntimeError if fails to send scan request within timeout.
        """
        # Check if radar is connected and not already collecting data
        if self.connected:
            
            # Check if scan count is within bounds
            if scan_count < STOP_SCAN_COUNT or scan_count > FOREVER_SCAN_COUNT:
                raise ValueError(('Requested number of scans {0} is outside valid range of {1} ' + 
                                  'and {2}').format(scan_count, STOP_SCAN_COUNT, 
                                       FOREVER_SCAN_COUNT))
            
            self.logger.info(('Requesting radar scan with {0} scans with scan interval ' + 
                              'of {1}...').format(scan_count, scan_interval))
            
            # Create scan request and send
            payload = {'scan_count': scan_count, 'scan_interval': scan_interval}
            message = self.encode_host_to_radar_message(payload, MRM_CONTROL_REQUEST)
            self.connection['sock'].sendto(message,
                           (self.connection['udp_ip_radar'], self.connection['udp_port_radar']))
            
            # Check if scan request was successful or not within timeout
            if scan_count != STOP_SCAN_COUNT:
                start = time.time()
                status_flag = -1
                while (time.time() - start) < self.settings['scan_request_timeout']:
                    try:
                        message, addr = self.connection['sock'].recvfrom(MAX_PACKET_SIZE)
                        payload = self.decode_radar_to_host_message(message, MRM_CONTROL_CONFIRM)
                        status_flag = payload['status']
                        break
                    except:
                        pass
                if status_flag == -1:
                    raise RuntimeError('Scan request timed out!')
                elif status_flag != 0:
                    raise RuntimeError(('Failed scan request with error code {0}!').format(
                            status_flag))
                self.logger.info('Scan request successful!')
                return status_flag
            
            else:
                self.logger.info(('Stop scan request made; no confirmation from radar will be ' +
                                  'provided!'))
                return 0
                
        else:
            raise RuntimeError('Radar not connected!')
            
    def read_scan_data(self, scan_data_filename=None, return_data=False, num_packets=None):
        """Read data returned from radar scans.
        
        Args:
            scan_data_filename (str)
                Path and name of file to save radar scans to. If None then data is not saved. 
                Defaults to None.
                
            return_data (bool)
                Flag indicating whether or not to return read data; flag exists to avoid creating 
                large internal variables when not needed. Defaults to False.
                
            num_packets (int)
                Number of packets to read. Appropriate value depends on the configuration of the 
                last scan reques. If None then packets will be read until stop flag is posted to 
                control file. Defaults to None.
                
        Returns:
            scan_data (bytes)
                Scan data read from the radar. Needs to unpacked to properly access scan 
                information. Will only be non-empty if return_data is set to True.
        
        Raises:
            RuntimeError if radar not connected already.
        """
        # Check if radar is connected and not already collecting data
        if self.connected:
            
            # Default return data
            scan_data = b''
            
            # Create scan data file if needed
            if scan_data_filename is not None:
                scan_data_file = open(scan_data_filename, 'wb')
                
                # Add all configuration values to save file
                config_bytes = self.config_to_bytes()
                scan_data_file.write(config_bytes)
            
            # Read fixed length or streaming data off radar
            self.logger.info('Reading data from the radar...')
            packet_count = 0
            start = time.time()
            while True:
                try:
                    packet_data, addr = self.connection['sock'].recvfrom(MAX_PACKET_SIZE)
                    if return_data:
                        scan_data += packet_data
                    if scan_data_filename is not None:
                        scan_data_file.write(packet_data)
                    packet_count += 1
                    start = time.time()
                    
                    # Read the specified number of packets
                    if num_packets is not None:
                        if packet_count == num_packets:
                            break
                    
                    # Read until stop flag has been posted to the control file
                    else:
                        self.control_file_handle.seek(0)
                        stop_flag = self.control_file_handle.read()
                        if stop_flag != '0':
                            self.scan_request(scan_count=STOP_SCAN_COUNT)
                            self.control_file_handle.close()
                            self.control_file_handle = open(CONTROL_FILENAME, 'w')
                            self.control_file_handle.write('0')
                            self.control_file_handle.close()
                            self.control_file_handle = open(CONTROL_FILENAME, 'r+')
                            break
                
                # Check if single packet read timeout threshold has been violated
                except:
                    if (time.time() - start) > self.settings['read_scan_data_timeout']:
                        raise RuntimeError('Radar scan data packet read timed out!')
   
            # Read any remaining streaming radar data
            if num_packets is not None:
                start = time.time()
                while (time.time() - start) < self.settings['read_residual_timeout']:
                    try:
                        packet_data, addr = self.connection['sock'].recvfrom(MAX_PACKET_SIZE)
                        if return_data:
                            scan_data += packet_data
                        if scan_data_filename is not None:
                           scan_data_file.write(packet_data)
                    except:
                        pass
            self.logger.info('Successfully read all the data!')
            
            # Close scan data file
            if scan_data_filename is not None:
                scan_data_file.close()
            return scan_data
        
        else:
            raise RuntimeError('Radar not connected!')
            
    def quick_look(self, scan_data_filename=None, return_data=False):
        """Executes quick-look with radar to confirm desired operation.
        
        Args:
            scan_data_filename (str)
                Path and name of file to save radar scans to. If None then data is not saved. 
                Defaults to None.
                
            return_data (bool)
                Flag indicating whether or not to return read data; flag exists to avoid creating 
                large internal variables when not needed. Defaults to False.
        
        Returns:
            scan_data (bytes)
                Scan data read from the radar. Needs to unpacked to properly access scan 
                information. Will only be non-empty if return_data is set to True.
        
        Raises:
            RuntimeError if scan data is not being either saved or returned.
            RuntimeError if radar not connected already or already collecting data.
        """
        # Check if data is being saved in some fashion
        if not return_data and not scan_data_filename:
            raise RuntimeError('Scan data not being saved to file or returned!')
        
        # Compute number of expected data packets in quick-look
        num_quick_look_packets = (math.ceil(float(self.N_bin) / SEG_NUM_BINS) *
                                  self.settings['quick_look_num_scans'])
        
        # Check if radar is connected and not already collecting data
        if self.connected and not self.collecting:
            self.logger.info('Starting quick-look mode...')
            
            # Send a scan request
            self.scan_request(self.settings['quick_look_num_scans'])
            self.collecting = True
            
            # Read streaming data from radar and save if desired
            scan_data = self.read_scan_data(scan_data_filename, return_data, num_quick_look_packets)
            self.collecting = False
            self.logger.info('Completed quick-look mode!')
            
        else:
            raise RuntimeError('Radar not connected or is already collecting data!')
        
        return scan_data
        
    def collect(self, scan_count=FOREVER_SCAN_COUNT, scan_interval=CONTINUOUS_SCAN_INTERVAL,
                scan_data_filename=None, return_data=False):
        """Collects radar data continuously until commanded to stop.
        
        Args:
            scan_count (int)
                Number of scans to collect. Defaults to FOREVER_SCAN_COUNT.
            
            scan_interval (int)
                Interval between sequential scans (us). Defaults to CONTINUOUS_SCAN_INTERVAL.
            
            scan_data_filename (str)
                Path and name of file to save radar scans to. If None then data is not saved. 
                Defaults to None.
                
            return_data (bool)
                Flag indicating whether or not to return read data; flag exists to avoid creating 
                large internal variables when not needed. Defaults to False.
                
        Returns:
            scan_data (bytes)
                Scan data read from the radar. Needs to unpacked to properly access scan 
                information. Will only be non-empty if return_data is set to True.
        
        Raises:
            ValueError if number of scans is less than minimum accepted value.
            RuntimeError if scan data is not being either saved or returned.
            RuntimeError if radar not connected already or already collecting data.
        """
        # Check if number of scans is less than minimum
        if scan_count < MIN_SCAN_COUNT:
            raise ValueError('Cannot request less than {0} scans!'.format(MIN_SCAN_COUNT))
        
        # Check if data is being saved in some fashion
        if not return_data and not scan_data_filename:
            raise RuntimeError('Scan data not being saved to file or returned!')
            
        # Check if radar is connected and not already collecting data
        if self.connected and not self.collecting:
            self.logger.info('Starting collect mode with {0} scans...'.format(scan_count))
            
            # Send a scan request
            self.scan_request(scan_count=scan_count, scan_interval=scan_interval)
            self.collecting = True
            
            # Read either undetermined amount of data from continous scanning or predetermined 
            # amount of scan data based on finite scan count
            num_packets = None
            if scan_count != FOREVER_SCAN_COUNT:
                num_packets = (math.ceil(float(self.N_bin) / SEG_NUM_BINS) * scan_count)
            scan_data = self.read_scan_data(scan_data_filename=scan_data_filename, 
                                            return_data=return_data, 
                                            num_packets=num_packets)
            self.collecting = False
            self.logger.info('Stopped collect mode!')
            
        else:
            raise RuntimeError('Radar not connected or is already collecting data!')
            
        return scan_data
        