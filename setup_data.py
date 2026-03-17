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

      # Try both possible zip names
      zip_path = os.path.join(script_dir, 'deliveries.zip')
      if not os.path.exists(zip_path):
          zip_path = os.path.join(script_dir, 'deliveries.csv.zip')
      csv_path = os.path.join(script_dir, 'deliveries.csv')

      if os.path.exists(csv_path):
          print("deliveries.csv already exists. Skipping extraction.")
          return

      if not os.path.exists(zip_path):
          print("ERROR: deliveries.zip not found!")
          print("Please make sure deliveries.zip is in the same folder as this script.")
          sys.exit(1)

      print("Extracting deliveries.csv from zip...")
      with zipfile.ZipFile(zip_path, 'r') as zf:
          # Extract the first CSV file found in the zip
          csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
          if csv_files:
              zf.extract(csv_files[0], script_dir)
              # Rename if needed
              extracted = os.path.join(script_dir, csv_files[0])
              if extracted != csv_path:
                  os.rename(extracted, csv_path)
          else:
              zf.extractall(script_dir)

      size_mb = os.path.getsize(csv_path) / (1024 * 1024)
      print(f"Done! deliveries.csv extracted ({size_mb:.1f} MB)")

  if __name__ == '__main__':
      main()
