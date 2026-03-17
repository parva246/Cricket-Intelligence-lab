IPL EXPERT PREDICTOR — SETUP INSTRUCTIONS
==========================================

STEP 1: Copy the entire "IPL_Expert_App" folder to your personal laptop
        (via RDP, USB, or however you transfer files)

STEP 2: Copy matches.csv and deliveries.csv into the IPL_Expert_App folder
        From: C:\Temp\matches.csv and C:\Temp\deliveries.csv
        To:   IPL_Expert_App\matches.csv and IPL_Expert_App\deliveries.csv

        Your folder should look like this:
        IPL_Expert_App/
            ipl_predictor.py
            matches.csv
            deliveries.csv
            README_SETUP.txt

STEP 3: Open Command Prompt on your laptop and run this ONE command:
        pip install streamlit xgboost pandas scikit-learn plotly seaborn matplotlib numpy

STEP 4: Navigate to the folder:
        cd C:\path\to\IPL_Expert_App

STEP 5: Run the app:
        streamlit run ipl_predictor.py

STEP 6: A browser tab opens automatically with your IPL Predictor app!
        (If it doesn't, go to http://localhost:8501 in your browser)

NOTE: First run takes a few minutes (training the model).
      After that it's cached and loads instantly.

TO STOP THE APP: Press Ctrl+C in the Command Prompt window.
TO RESTART: Just run "streamlit run ipl_predictor.py" again.
