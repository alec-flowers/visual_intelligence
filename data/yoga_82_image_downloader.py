import http
import urllib.error
import os
import wget
import pandas as pd

"""
Small script to download the images of the Yoga.82 data set. Required file structure:
Good Yoga/
|
---------pose_to_download_dir/  # needs to be created by user
|
---------pose_to_download.txt
"""

# put pose_to_download.txt into os.path.join of next line
df = pd.read_csv(os.path.join("Good Yoga", "Warrior_II_Pose_or_Virabhadrasana_II_.txt"), sep='\t', header=None)
df.columns = ['image_name', 'link']
output_directory = "Good Yoga"
not_downloaded_counter = 0

for idx, row in df.iterrows():
    try:
        wget.download(row['link'], out=os.path.join(output_directory, row['image_name']))
    except urllib.error.HTTPError as e:
        not_downloaded_counter += 1
    except urllib.error.URLError as e:
        not_downloaded_counter += 1
    except http.client.RemoteDisconnected as e:
        not_downloaded_counter += 1

print(f"Failed to download {not_downloaded_counter} out of {len(df)} images.")
