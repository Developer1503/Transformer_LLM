# scripts/scrape_data.py

import requests
import re
from bs4 import BeautifulSoup
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

URL = "https://en.wikipedia.org/wiki/Natural_language_processing"
FILE_PATH = "data/corpus.txt"

try:
    resp = requests.get(URL)
    resp.raise_for_status()  # Raise an exception for bad status codes
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Extract text from all paragraph tags
    text = " ".join(p.get_text() for p in soup.find_all("p"))
    
    # Clean the text
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation marks like [1]
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.write(text)
        
    print(f"✅ Data successfully scraped and saved to {FILE_PATH}")

except requests.exceptions.RequestException as e:
    print(f"❌ Error scraping data: {e}")