#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Helper functions for UAS-SAR."""

__author__ = "Ramamurthy Bhagavatula"
__version__ = "1.0"
__maintainer__ = "Ramamurthy Bhagavatula"
__email__ = "ramamurthy.bhagavatula@ll.mit.edu"

# Import required modules and methods
import os
import sys
import numpy as np
from warnings import warn
from pprint import pprint

def value_to_message(value, num_bytes):
    """
    Converts a integer configuration value to a message compatible byte 
    (character) format to be appended to set configuration request.
    Inputs:
        value - Integer value.
        num_bytes - The number of bytes to package the integer into.
    Outputs:
        message - A byte/character formatted sequence representing the 
                  input value.
    """
    num_nibbles = 2 * num_bytes
    hex_rep = hex(int(value))[2:]
    hex_rep = ('0' * (num_nibbles - len(hex_rep)) + hex_rep)
    message = ''
    for ii in range(0, num_nibbles, 2):
        message += chr(int(hex_rep[ii:(ii + 2)], 16))
    return message

def verbose_print(verbose, message):
    """
    Prints message only if verbosity flag is set.
    Inputs:
        verbose - Boolean flag indicating whether or not to print message.
        message - String containing message to print.
    Outputs:
        None
    """
    if verbose:
        if isinstance(message, dict):
            pprint(message)
        else:
            print(message)

def verbose_progress_bar(verbose, step, total, message=''):
    """
    Prints/updates console text progress bar only if verbosity flag is set.
    Inputs:
        verbose - Boolean flag indicating whether or not to display progress
                  bar.
        count - Current step number.
        total - Total number of steps.
        message - String containing message to print.
    Outputs:
        None
    """
    # Only display if requested
    if verbose:
        
        # Display constants
        complete_symbol = '='
        incomplete_symbol = ' '
        total_len = 50
        
        # Compute current completion
        complete_len = int(round(total_len * step / float(total)))
        complete_percent = round(100.0 * step / float(total), 1)
        progress_bar = (complete_symbol * complete_len + incomplete_symbol * 
                        (total_len - complete_len))
        
        # Print progress bar
        sys.stdout.write('%s...[%s] %.1f%%\r' % 
                         (message, progress_bar, complete_percent))
        sys.stdout.flush()

def find_new_file(filename):
    """
    Generates a new filename by appending a number to the end of the input one;
    assumes input file already exists.
    Inputs:
        filename - Path and name of file to use as base.
    Outputs:
        new_file - Path and name of new file.
    """
    if not os.path.exists(filename):
        return filename
    else:
        name, ext = os.path.splitext(filename)
        ii = 0
        while os.path.exists('%s_%d%s' % (name, ii, ext)):
            ii += 1
        return '%s_%d%s' % (name, ii, ext)

def yes_or_no(question):
    """
    Prompts user to answer a yes or no question through keyboard input.
    Inputs:
        question - String containing question to ask.
    Outputs:
        answer - Boolean where True indicates that yes was selected and where
                 False indicates that no was selected.
    """
    # Set of acceptable answers
    yes = set(['yes', 'y'])
    no = set(['no', 'n'])
    
    # Iterate till answer is given
    while True:
        answer = input(question + "(y/n): ").lower().strip()
        if answer in yes:
            return True
        elif answer in no:
            return False
        else:
            print("Please respnd with 'yes' or 'no'")

def is_valid_file(parser, filename, mode):
    """
    Check if specified argument is a valid file for read or write; meant to be
    used in the context of argparse.ArgumentParser.
    Inputs:
        parser - arparse.ArgumentParser instance.
        filename - File to check.
        mode - The read/write mode to check the file for; options are 'r' and 
               'w'.
    Outputs:
        None
    """
    if filename:
        if mode == 'r':
            try:
                f = open(filename, 'r')
                f.close()
            except:
                parser.error("The file %s does not exist or cannot be read!" % 
                             filename)
        elif mode == 'w':
            try:
                f = open(filename, 'w')
                f.close()
                os.remove(filename)
            except:
                parser.error("The file %s does not exist or cannot be " +
                             "written to!" % filename)
        else:
            parser.error("Unrecognized file mode specified!")

def linear_interp_nan(coords, vals):
    """
    Linear 1-D interpolation of data that may have missing values and/or 
    coordinates. Assumes that coordinates are uniformly spaced.
    Inputs:
        coords - Array-like vector of size M representing data coordinates; may
                 have NaNs in it.
        vals - Array-like matrix containing M x N elements representing data 
               values; may have NaNs in it.
    Outputs:
        coords_out - Size M numpy array representing data coordinates with NaNs
                     replaced; should match input coords dimensionality.
        vals_out - numpy matrix representing data values with NaNs replaced; 
                   should match input vals dimensionality.
    """
    # Initialize outputs; make a deep copy to ensure that inputs are not 
    # directly modified
    coords_out = np.copy(coords)
    vals_out = np.copy(vals)
    
    # Store inputs original shapes
    coords_shape = coords_out.shape
    vals_shape = vals_out.shape
    
    # Remove singleton dimensions from coordinates
    coords_out = np.squeeze(coords_out)
    
    # Check inputs
    if coords_out.ndim != 1:
        raise ValueError('Coordinates are not 1-D!')
        
    if vals_out.ndim > 2:
        raise ValueError('Data must be a 2-D matrix!')
    elif vals_out.ndim == 1:
        vals_out = np.reshape(vals_out, (-1, 1))
        
    dim_match = coords_out.size == np.asarray(vals_shape)
    transpose_flag = False
    if not np.any(dim_match):
        raise IndexError('No apparent dimension agreement between ' + 
                         'coordinates and data!')
    elif np.all(dim_match):
        warn(('Ambiguous dimensionalities; assuming columns of data are to ' + 
              'be interpolated'), Warning)
    elif dim_match[0] != 1:
        vals_out = vals_out.transpose()
        transpose_flag = True
        
    # Determine where NaN coordinates are replace them using linear 
    # interpolation assuming uniform spacing
    uniform_spacing = np.arange(0, coords_out.size)
    coords_nan = np.isnan(coords_out)
    coords_out[coords_nan] = np.interp(uniform_spacing[coords_nan], 
          uniform_spacing[~coords_nan], coords_out[~coords_nan])
    
    # Iterate over each dimension of data
    for ii in range(0, vals_shape[1]):
        
        # Determine where the NaN data and replace them using linear 
        # interpolation
        vals_nan = np.isnan(vals_out[:, ii])
        vals_out[vals_nan, ii] = np.interp(coords_out[vals_nan], 
                coords_out[~vals_nan], vals_out[~vals_nan, ii])
        
    # Reshape results to match inputs
    coords_out = np.reshape(coords_out, coords_shape)
    if transpose_flag:
        vals_out = np.transpose(vals_out)
    vals_out = np.reshape(vals_out, vals_shape)
    
    # Return coordinates and data with NaN values replaced
    return coords_out, vals_out