import os
import requests
import zipfile

# Define the download directory
download_dir = "data"

# Create the directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Define the files to download
files_to_download = [
    {
        "name": "htr-selex.zip",
        "url": "https://www.dropbox.com/scl/fi/yoavlu26hxq4i7liyhof2/htr-selex.zip?rlkey=vlu7xlfx6o3q2gnk2b97qemla&st=c53p0zfo&dl=0"
    },
    {
        "name": "RNAcompete_intensities.zip",
        "url": "https://www.dropbox.com/scl/fi/t1l9k156k6692d7ad0c8d/RNAcompete_intensities.zip?rlkey=qlk6jlog3pdbs3l97ki9jpqjo&st=xugqm5w2&dl=0"
    },
    {
        "name": "RNAcompete_sequences.txt",
        "url": "https://www.dropbox.com/scl/fi/zlvuw20c6weatrtz294kn/RNAcompete_sequences.txt?rlkey=qbqtt4jb141vz6atskq93s2il&st=p1dzdiv3&dl=0"
    }
]

# Function to download files from Dropbox
def download_file_from_dropbox(url, destination):
    # Modify the URL to get the direct download link
    direct_download_url = url.replace('?dl=0', '?dl=1').replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    print(f"Downloading from {direct_download_url} to {destination}...")
    
    try:
        response = requests.get(direct_download_url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {destination}")
    except requests.RequestException as e:
        print(f"Failed to download {destination}: {e}")

# Download each file
for file in files_to_download:
    output_path = os.path.join(download_dir, file['name'])
    print(f"Downloading {file['name']}...")
    download_file_from_dropbox(file['url'], output_path)

    # Check the downloaded file size
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Downloaded {output_path} with size {file_size} bytes.")
    else:
        print(f"File {output_path} does not exist after download.")

# Function to extract zip files into their dedicated folders and remove them
def extract_and_cleanup_zip_files(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if item_path.endswith('.zip'):
            if zipfile.is_zipfile(item_path):
                extract_dir = os.path.join(directory, os.path.splitext(item)[0])
                os.makedirs(extract_dir, exist_ok=True)
                print(f"Extracting {item} to {extract_dir}...")
                try:
                    with zipfile.ZipFile(item_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    os.remove(item_path)
                    print(f"Deleted {item} after extraction.")
                except zipfile.BadZipFile:
                    print(f"Failed to extract {item}: Bad zip file.")
            else:
                print(f"Skipping extraction of {item}: Not a valid zip file.")
        else:
            print(f"Skipping {item} as it's not a zip file.")

# Extract any downloaded zip files and clean up
extract_and_cleanup_zip_files(download_dir)

print("Download, extraction, and cleanup completed.")
