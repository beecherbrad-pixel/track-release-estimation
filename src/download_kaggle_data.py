import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Load environment variables from the .env file in the root directory
# Since this script is in /src, we look one level up
load_dotenv(dotenv_path="../.env")

def download_kaggle_data(kaggle_dataset, data_path="../data"):
    # 1. Authenticate using the environment variables
    # The Kaggle API automatically looks for KAGGLE_USERNAME and KAGGLE_KEY
    try:
        api = KaggleApi()
        api.authenticate()
        print("Authenticated with Kaggle.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    # 2. Define dataset and destination
    dataset = kaggle_dataset
    data_path = data_path
    
    # Create the data folder if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # 3. Download and unzip
    print(f"Downloading {dataset} to {data_path}...")
    api.dataset_download_files(dataset, path=data_path, unzip=True)
    print("Download complete! You can now access the CSVs in the /data folder.")
