#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""u-blox message formats."""

__author__ = "Ramamurthy Bhagavatula"
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
import numpy as np
from collections import OrderedDict

# Packet definition taken from [1] (section 32.18.14.1 Navigation Position 
# Velocity Time Solution, pages 307 - 309)
UBX_SYNC1 = b'\xB5'
UBX_SYNC2 = b'\x62'
MESSAGE_CLASS = b'\x01'
MESSAGE_ID = b'\x07'
PAYLOAD_LEN = 92 # bytes
BITFIELD_VALID = OrderedDict([ # Bits of valid field from MSB to LSB; sub-fieldname: [number of bits, signed]
        ('reservedValid', [4, False]), # Reserved
        ('validMag', [1, False]), # Valid magnetic declination
        ('fullyResolved', [1, False]), # UTC time of day has been fully resolved; no seconds uncertainty
        ('validTime', [1, False]), # Valid UTC time
        ('validDate', [1, False])]) # Valid UTC date
BITFIELD_FLAGS = OrderedDict([ # Bits of flags field
        ('carrSoln', [2, False]), # Carrier phase range solution status
        ('headVehValid', [1, False]), # Heading of vehicle is valid
        ('psmState', [3, False]), # Power save mode
        ('diffSoln', [1, False]), # Differential corrections were applied
        ('gnssFixOK', [1, False])]) # Valid fix
BITFIELD_FLAGS2 = OrderedDict([ # Bits of flags2 field
        ('confirmedTime', [1, False]), # UTC time of day could be confirmed
        ('confirmedDat', [1, False]), # UTC date validity could be confirmed
        ('confirmedAvai', [1, False]), # Information about UTC Date and Time of Day validity confirmation is available
        ('reservedFlags2', [5, False])]) # Reserved
PAYLOAD_FORMAT = OrderedDict([ # Payload format; fieldname: [number format, count, scaling, bitfield]
    ('iTOW', [np.dtype(np.uint32), 1, 1, None]), # GPS time (ms)
    ('year', [np.dtype(np.uint16), 1, 1, None]), # {UTC}
    ('month', [np.dtype(np.uint8), 1, 1, None]), # [1 to 31] {UTC}
    ('day', [np.dtype(np.uint8), 1, 1, None]), # [1 to 12] {UTC}
    ('hour', [np.dtype(np.uint8), 1, 1, None]), # [0 to 23] {UTC}
    ('min', [np.dtype(np.uint8), 1, 1, None]), # [0 to 59] {UTC}
    ('sec', [np.dtype(np.uint8), 1, 1, None]), # [0 to 60] {UTC}
    ('valid', [np.dtype(np.uint8), 1, 1, BITFIELD_VALID]), # Validity flags
    ('tAcc', [np.dtype(np.uint32), 1, 1, None]), # Time accuracy estimate {UTC} (ns)
    ('nano', [np.dtype(np.int32), 1, 1, None]), # Fraction of second [-1e-9 to 1e9] {UTC} (ns)
    ('fixType', [np.dtype(np.uint8), 1, 1, None]), # GNSS fix type [0 to 5]
    ('flags', [np.dtype(np.uint8), 1, 1, BITFIELD_FLAGS]), # Fix status flags
    ('flags2', [np.dtype(np.uint8), 1, 1, BITFIELD_FLAGS2]), # Additional flags
    ('numSV', [np.dtype(np.uint8), 1, 1, None]), # Number of satellites used in NAV solution
    ('lon', [np.dtype(np.int32), 1, 1e-7, None]), # Longitude (deg)
    ('lat', [np.dtype(np.int32), 1, 1e-7, None]), # Latitude (deg)
    ('height', [np.dtype(np.int32), 1, 1, None]), # Height above ellipsoid (mm)
    ('hMSL', [np.dtype(np.int32), 1, 1, None]), # Height above mean sea level (mm)
    ('hAcc', [np.dtype(np.uint32), 1, 1, None]), # Horizontal accuracy estimate (mm)
    ('vAcc', [np.dtype(np.uint32), 1, 1, None]), # Vertical accuracy estimate (mm)
    ('velN', [np.dtype(np.int32), 1, 1, None]), # NED north velocity (mm/s)
    ('velE', [np.dtype(np.int32), 1, 1, None]), # NED east velocity (mm/s)
    ('velD', [np.dtype(np.int32), 1, 1, None]), # NED down velocity (mm/s)
    ('gSpeed', [np.dtype(np.int32), 1, 1, None]), # Ground speed {2-D} (mm/s)
    ('headMot', [np.dtype(np.int32), 1, 1e-5, None]), # Heading of motion {2-D} (deg)
    ('sAcc', [np.dtype(np.uint32), 1, 1, None]), # Speed accuracy estimate (mm/s)
    ('headAcc', [np.dtype(np.uint32), 1, 1e-5, None]), # Heading accuracy estimate; both motion and vehicle (deg)
    ('pDOP', [np.dtype(np.uint16), 1, 0.01, None]), # Position DOP
    ('reserved1', [np.dtype(np.uint8), 6, 1, None]), # Reserved
    ('headVeh', [np.dtype(np.int32), 1, 1e-5, None]), # Heading of vehicle {2-D} (deg)
    ('magDec', [np.dtype(np.int16), 1, 1e-2, None]), # Magnetic declination (deg)
    ('magAcc', [np.dtype(np.uint16), 1, 1e-2, None])]) # Magnetic declination accuracy (deg)
