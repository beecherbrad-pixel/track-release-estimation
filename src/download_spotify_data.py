import os
import time
import pandas as pd
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from tqdm.notebook import tqdm

def get_release_dates(track_ids):
    # Ensure the .env is loaded specifically for this function call
    load_dotenv('../.env')

    # Calculate total batches for the progress bar
    total_batches = range(0, len(track_ids), 50)

    # Re-initialize the auth manager inside the function
    auth_manager = SpotifyClientCredentials(
        client_id=os.getenv('SPOTIPY_CLIENT_ID'),
        client_secret=os.getenv('SPOTIPY_CLIENT_SECRET')
    )
    sp = Spotify(auth_manager=auth_manager)
    
    release_data = []
    
    # Process in batches
    for i in tqdm(total_batches, desc="Fetching Spotify Dates"):
        batch = track_ids[i:i+50]
        for track in batch:
            try:
                track_data = sp.track(f'spotify:track:{track}')
                release_data.append({
                    'track_id': track,
                    'release_date': track_data['album']['release_date'],
                })
                #print(f"Successfully obtained {track_data['name']}")
            except Exception as e:
                tqdm.write(f"Error in batch {i//50} for track {track}: {e}")
                continue
        # Short sleep to avoid the 429 "Too Many Requests" error
        time.sleep(1.0) 

    return pd.DataFrame(release_data)