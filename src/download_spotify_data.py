import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

# Initialize the API
auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_release_dates(track_ids):
    release_data = []
    
    # Spotify allows max 50 IDs per request
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i:i+50]
        results = sp.tracks(batch)
        
        for track in results['tracks']:
            if track:
                release_data.append({
                    'track_id': track['id'],
                    'release_date': track['album']['release_date'],
                    'release_date_precision': track['album']['release_date_precision']
                })
        
        # Respect the API; short sleep to avoid 429 errors
        time.sleep(0.1) 
        
    return pd.DataFrame(release_data)

