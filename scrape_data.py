import os
import requests
from bs4 import BeautifulSoup
import json

# URL of the website to scrape
url = "https://example.com/gallery"

# Headers to mimic a real browser request
headers = {"User-Agent": "Mozilla/5.0"}

# Send a GET request
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Create a folder to save images
os.makedirs("scraped_images", exist_ok=True)

# Find all image tags
images = soup.find_all("img")

# List to store image paths and captions
data = []

for img in images:
    img_url = img["src"]
    caption = img.get("alt", "No Caption")  # Extract alt-text as caption

    # Download the image
    img_data = requests.get(img_url).content
    img_name = os.path.join("scraped_images", img_url.split("/")[-1])

    with open(img_name, "wb") as f:
        f.write(img_data)

    # Store the image path and caption
    data.append({"image_path": img_name, "caption": caption})

    print(f"Saved {img_name} with caption: {caption}")

# Save the data to a JSON file
with open("scraped_data.json", "w") as f:
    json.dump(data, f)
