#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Time-Based Anomaly Classifier

This script processes a folder of images with timestamps in their filenames
and classifies them as 'normal' or 'anomaly' based on user-defined time ranges.

Format of image filenames: dataset_image_YYYYMMDD-HHMMSS.jpg
Example: dataset_image_20250408-082552.jpg
"""

import os
import re
import shutil
import argparse
from datetime import datetime, time


def extract_datetime(filename):
    """
    Extract date and time from filename.
    
    Args:
        filename (str): Image filename in format dataset_image_YYYYMMDD-HHMMSS.jpg
        
    Returns:
        datetime: Extracted datetime object or None if pattern doesn't match
    """
    pattern = r'dataset_image_(\d{8})-(\d{6})'
    match = re.search(pattern, filename)
    
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        return datetime(year, month, day, hour, minute, second)
    
    return None


def is_anomaly(dt, anomaly_time_ranges):
    """
    Check if a datetime falls within any of the specified anomaly time ranges.
    
    Args:
        dt (datetime): The datetime to check
        anomaly_time_ranges (list): List of (start_time, end_time) tuples
        
    Returns:
        bool: True if the time is within any anomaly range, False otherwise
    """
    current_time = dt.time()
    
    for start_time, end_time in anomaly_time_ranges:
        if start_time <= current_time <= end_time:
            return True
    
    return False


def classify_images(input_folder, output_folder, anomaly_time_ranges):
    """
    Classify images as 'normal' or 'anomaly' based on their timestamps.
    
    Args:
        input_folder (str): Path to the folder containing images
        output_folder (str): Path to create output classification folders
        anomaly_time_ranges (list): List of (start_time, end_time) tuples defining anomaly periods
    """
    # Create output directories
    normal_dir = os.path.join(output_folder, 'normal')
    anomaly_dir = os.path.join(output_folder, 'anomaly')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        file_path = os.path.join(input_folder, filename)
        
        # Extract datetime from filename
        dt = extract_datetime(filename)
        if not dt:
            print(f"Warning: Unable to extract datetime from {filename}, skipping...")
            continue
        
        # Determine classification
        if is_anomaly(dt, anomaly_time_ranges):
            dest_dir = anomaly_dir
            classification = "anomaly"
        else:
            dest_dir = normal_dir
            classification = "normal"
        
        # Copy the file
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(file_path, dest_path)
        print(f"Classified {filename} as {classification}")


def parse_time_range(time_range_str):
    """
    Parse a time range string in format 'HH:MM:SS-HH:MM:SS'.
    
    Args:
        time_range_str (str): String representing a time range
        
    Returns:
        tuple: (start_time, end_time) as time objects
    """
    start_str, end_str = time_range_str.split('-')
    
    def parse_time(time_str):
        hours, minutes, seconds = map(int, time_str.split(':'))
        return time(hours, minutes, seconds)
    
    return parse_time(start_str), parse_time(end_str)


def main():
    # CONFIGURE YOUR SETTINGS HERE
    # ===========================
    
    # Path to your folder containing the images
    input_folder = "C:/Juanjo/Veolia/sludge_image_viewer/sludge_image_viewer/local_dataset_experiment_20250408"
    
    # Path where you want the output folders (normal/anomaly) to be created
    output_folder = "./data/raw"
    
    # Define your anomaly time ranges here based on your observations:
    # - "Looks good", "Looks really good", or "Looks okay/passable" = NORMAL
    # - Everything else = ANOMALY
    anomaly_time_ranges_str = [
        "09:09:00-09:28:00",  # 9:10 "Looks a bit thin" to 9:35 "Looks good"
        "09:54:00-10:14:00",  # 9:50 "Looks a bit thin" to 10:15 "Looks okay"
        "10:22:00-10:39:59",  # 10:25 "Looks a bit thin" to 10:40 "Looks good"
    ]
    
    # ===========================
    # END OF CONFIGURATION
    
    # Parse time ranges
    anomaly_time_ranges = [parse_time_range(time_range) for time_range in anomaly_time_ranges_str]
    
    # Run classification
    classify_images(input_folder, output_folder, anomaly_time_ranges)
    
    print(f"Classification complete. Results saved to {output_folder}")


if __name__ == '__main__':
    main()