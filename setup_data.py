"""
IPL Match Predictor — Data Setup Script
========================================
Run this script ONCE after cloning the repo to extract the data files.

Usage:
    python setup_data.py
"""

import zipfile
import os
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Extract deliveries.csv from zip
    zip_path = os.path.join(script_dir, 'deliveries.csv.zip')
    csv_path = os.path.join(script_dir, 'deliveries.csv')

    if os.path.exists(csv_path):
        print("deliveries.csv already exists. Skipping extraction.")
        return

    if not os.path.exists(zip_path):
        print("ERROR: deliveries.csv.zip not found!")
        print("Please make sure deliveries.csv.zip is in the same folder as this script.")
        sys.exit(1)

    print("Extracting deliveries.csv from zip...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract('deliveries.csv', script_dir)

    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"Done! deliveries.csv extracted ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
