import pandas as pd
import os

def log_training_results(file_path, session_data):
    """
    Logs the training session details into a CSV file.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    - session_data (dict): A dictionary containing all the session details.
                           Keys should match the column names.
    """
    # Convert the session data into a DataFrame
    session_df = pd.DataFrame([session_data])
    
    if not os.path.exists(file_path):
        # If file doesn't exist, create it with headers
        session_df.to_csv(file_path, index=False)
    else:
        # If file exists, append the new data
        session_df.to_csv(file_path, mode='a', header=False, index=False)
