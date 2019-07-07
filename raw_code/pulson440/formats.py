#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PulsON 440 message formats."""

__author__ = "Ramamurthy Bhagavatula"
__version__ = "1.0"
__maintainer__ = "Ramamurthy Bhagavatula"
__email__ = "ramamurthy.bhagavatula@ll.mit.edu"

"""References
[1] Monostatic Radar Application Programming Interface (API) Specification
    PulsON (R) 400 Series
    Version: 1.2.2
    Date: January 2015
    https://timedomain.com/wp-content/uploads/2015/12/320-0298E-MRM-API-Specification.pdf
"""

# Import required modules and methods
from collections import OrderedDict
import numpy as np
from pulson440.constants import REC_ANTENNA_MODE, REC_PERSIST_FLAG, REC_SCAN_RES, RESERVED_VAL, \
    NOT_IMPLEMENTED_VAL

# Formats of various messages between host and radar. Each one is defined by a message type and a 
# packet definition. A packet definition is an order dictionary specifying the order of the packet
# fields. The values in these dictionaries depend on whether the message is for host to radar 
# messages or for radar to host messages.
# 
# For host to radar messages each key's value is a 2 element list where the first element is the 
# data type and the second value is the default value. If the default value is None then this part 
# of the packet must be user defined otherwise the default value is used.
#
# For radar to host messages each key's value is the data type. This difference in format is to 
# ensure the right message format is used for the right direction of communication.

# Set radar configuration request; host to radar
MRM_SET_CONFIG_REQUEST = {'message_type': 4097, # Message type
                          'packet_def': OrderedDict([
                                  ('message_type', [np.dtype(np.uint16), None]), # Message type
                                  ('message_id', [np.dtype(np.uint16), None]), # Message ID
                                  ('node_id', [np.dtype(np.uint32), None]), # Node ID
                                  ('scan_start', [np.dtype(np.int32), None]), # Scan start time (ps)
                                  ('scan_stop', [np.dtype(np.int32), None]), # Scan stop time (ps)
                                  ('scan_res', [np.dtype(np.uint16), REC_SCAN_RES]), # Scan resolution (bins); recommended value used
                                  ('pii', [np.dtype(np.uint16), None]), # Pulse integration index
                                  ('seg_1_samp', [np.dtype(np.uint16), NOT_IMPLEMENTED_VAL]), # Segment 1 samples; not used
                                  ('seg_2_samp', [np.dtype(np.uint16), NOT_IMPLEMENTED_VAL]), # Segment 2 samples; not used
                                  ('seg_3_samp', [np.dtype(np.uint16), NOT_IMPLEMENTED_VAL]), # Segment 3 samples; not used
                                  ('seg_4_samp', [np.dtype(np.uint16), NOT_IMPLEMENTED_VAL]), # Segment 4 samples; not used
                                  ('seg_1_int', [np.dtype(np.uint8), NOT_IMPLEMENTED_VAL]), # Segment 1 integration; not used
                                  ('seg_2_int', [np.dtype(np.uint8), NOT_IMPLEMENTED_VAL]), # Segment 2 integration; not used
                                  ('seg_3_int', [np.dtype(np.uint8), NOT_IMPLEMENTED_VAL]), # Segment 3 integration; not used
                                  ('seg_4_int', [np.dtype(np.uint8), NOT_IMPLEMENTED_VAL]), # Segment 4 integration; not used
                                  ('ant_mode', [np.dtype(np.uint8), REC_ANTENNA_MODE]), # Antenna mode; recommended value used
                                  ('tx_gain_ind', [np.dtype(np.uint8), None]), # Transmit gain index
                                  ('code_channel', [np.dtype(np.uint8), None]), # Code channel
                                  ('persist_flag', [np.dtype(np.uint8), REC_PERSIST_FLAG])])} # Persist flag
MRM_SET_CONFIG_REQUEST['packet_length'] = sum( # Packet length (bytes))
        [value[0].itemsize for value in MRM_SET_CONFIG_REQUEST['packet_def'].values()])

# Set radar configuration confirmation; radar to host
MRM_SET_CONFIG_CONFIRM = {'message_type': 4353, # Message type
                          'packet_def': OrderedDict([
                                  ('message_type', np.dtype(np.uint16)), # Message type
                                  ('message_id', np.dtype(np.uint16)), # Message ID
                                  ('status', np.dtype(np.uint32))])} # Set configuration status
MRM_SET_CONFIG_CONFIRM['packet_length'] = sum( # Packet length (bytes))
        [value.itemsize for value in MRM_SET_CONFIG_CONFIRM['packet_def'].values()])

# Get radar configuration request; host to radar
MRM_GET_CONFIG_REQUEST = {'message_type': 4098, # Message type
                          'packet_def': OrderedDict([
                                  ('message_type', [np.dtype(np.uint16), None]), # Message type
                                  ('message_id', [np.dtype(np.uint16), None])])} # Message ID
MRM_GET_CONFIG_REQUEST['packet_length'] = sum( # Packet length (bytes))
        [value[0].itemsize for value in MRM_GET_CONFIG_REQUEST['packet_def'].values()])

# Set radar configuration request; radar to host
MRM_GET_CONFIG_CONFIRM = {'message_type': 4354, # Message type
                          'packet_def': OrderedDict([
                                  ('message_type', np.dtype(np.uint16)), # Message type
                                  ('message_id', np.dtype(np.uint16)), # Message ID
                                  ('node_id', np.dtype(np.uint32)), # Node ID
                                  ('scan_start', np.dtype(np.int32)), # Scan start time (ps)
                                  ('scan_stop', np.dtype(np.int32)), # Scan stop time (ps)
                                  ('scan_res', np.dtype(np.uint16)), # Scan resolution (bins); recommended value used
                                  ('pii', np.dtype(np.uint16)), # Pulse integration index
                                  ('seg_1_samp', np.dtype(np.uint16)), # Segment 1 samples; not used
                                  ('seg_2_samp', np.dtype(np.uint16)), # Segment 2 samples; not used
                                  ('seg_3_samp', np.dtype(np.uint16)), # Segment 3 samples; not used
                                  ('seg_4_samp', np.dtype(np.uint16)), # Segment 4 samples; not used
                                  ('seg_1_int', np.dtype(np.uint8)), # Segment 1 integration; not used
                                  ('seg_2_int', np.dtype(np.uint8)), # Segment 2 integration; not used
                                  ('seg_3_int', np.dtype(np.uint8)), # Segment 3 integration; not used
                                  ('seg_4_int', np.dtype(np.uint8)), # Segment 4 integration; not used
                                  ('ant_mode', np.dtype(np.uint8)), # Antenna mode; recommended value used
                                  ('tx_gain_ind', np.dtype(np.uint8)), # Transmit gain index
                                  ('code_channel', np.dtype(np.uint8)), # Code channel
                                  ('persist_flag', np.dtype(np.uint8)), # Persist flag
                                  ('timestamp', np.dtype(np.uint32)), # Time since boot (ms)
                                  ('status', np.dtype(np.uint32))])} # Status
MRM_GET_CONFIG_CONFIRM['packet_length'] = sum( # Packet length (bytes))
        [value.itemsize for value in MRM_GET_CONFIG_CONFIRM['packet_def'].values()])

# Radar scan request; host to radar
MRM_CONTROL_REQUEST = {'message_type': 4099, # Message type
                       'packet_def': OrderedDict([
                               ('message_type', [np.dtype(np.uint16), None]), # Message type
                               ('message_id', [np.dtype(np.uint16), None]), # Message ID
                               ('scan_count', [np.dtype(np.uint16), None]), # Scan count
                               ('reserved', [np.dtype(np.uint16), RESERVED_VAL]), # Reserved
                               ('scan_interval', [np.dtype(np.uint32), None])])} # Scan interval (us)
MRM_CONTROL_REQUEST['packet_length'] = sum( # Packet length (bytes))
        [value[0].itemsize for value in MRM_CONTROL_REQUEST['packet_def'].values()])

# Radar scan confirm; radar to host
MRM_CONTROL_CONFIRM = {'message_type': 4355, # Message type
                       'packet_def': OrderedDict([
                               ('message_type', np.dtype(np.uint16)), # Message type
                               ('message_id', np.dtype(np.uint16)), # Message ID
                               ('status', np.dtype(np.uint32))])} # Status 
MRM_CONTROL_CONFIRM['packet_length'] = sum( # Packet length (bytes))
        [value.itemsize for value in MRM_CONTROL_CONFIRM['packet_def'].values()])

# Radar reboot request; host to radar
MRM_REBOOT_REQUEST = {'message_type': 61442, # Message type
                      'packet_def': OrderedDict([
                              ('message_type', [np.dtype(np.uint16), None]), # Message type
                              ('message_id', [np.dtype(np.uint16), None])])} # Message ID
MRM_REBOOT_REQUEST['packet_length'] = sum( # Packet length (bytes))
        [value[0].itemsize for value in MRM_REBOOT_REQUEST['packet_def'].values()])

# Radar reboot confirm; radar to host
MRM_REBOOT_CONFIRM = {'message_type': 61698, # Message type
                      'packet_def': OrderedDict([
                              ('message_type', np.dtype(np.uint16)), # Message type
                              ('message_id', np.dtype(np.uint16))])} # Message ID
MRM_REBOOT_CONFIRM['packet_length'] = sum( # Packet length (bytes))
        [value.itemsize for value in MRM_REBOOT_CONFIRM['packet_def'].values()])

# Scan data; radar to host
MRM_SCAN_INFO = {'message_type': 61953, # Message type
                 'packet_def': OrderedDict([
                         ('message_type', np.dtype(np.uint16)), # Message type
                         ('message_id', np.dtype(np.uint16)), # Message ID
                         ('node_id', np.dtype(np.uint32)), # Node ID
                         ('timestamp', np.dtype(np.uint32)), # Time since boot (ms)
                         ('reserved0', np.dtype(np.uint32)), # Reserved
                         ('reserved1', np.dtype(np.uint32)), # Reserved
                         ('reserved2', np.dtype(np.uint32)), # Reserved
                         ('reserved3', np.dtype(np.uint32)), # Reserved
                         ('scan_start', np.dtype(np.int32)), # Scan start time (ps)
                         ('scan_stop', np.dtype(np.int32)), # Scan stop time (ps)
                         ('scan_res', np.dtype(np.int16)), # Scan resolution (bins)
                         ('scan_type', np.dtype(np.uint8)), # Type of scan data
                         ('reserved4', np.dtype(np.uint8)), # Reserved
                         ('antenna_id', np.dtype(np.uint8)), # Receiving antenna designator
                         ('operational_mode', np.dtype(np.uint8)), # Operational mode
                         ('num_samples_message', np.dtype(np.uint16)), # Number of samples in this message
                         ('num_samples_total', np.dtype(np.uint32)), # Number of samples in single scan
                         ('message_index', np.dtype(np.uint16)), # Index of this message's portion of data in single scan
                         ('num_messages_total', np.dtype(np.uint16)), # Number of data messages in single scan
                         ('scan_data', np.dtype(np.int32))])} # Scan data
MRM_SCAN_INFO['packet_length'] = sum( # Packet length (bytes))
        [value.itemsize for value in MRM_SCAN_INFO['packet_def'].values()])