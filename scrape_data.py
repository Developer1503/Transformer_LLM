import os
import requests
from bs4 import BeautifulSoup
import json

# URL list of web pages to scrape
urls = [
    "https://example.com/news",
    "https://example.com/articles",
]

headers = {"User-Agent": "Mozilla/5.0"}
data = []

for url in urls:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract text data (modify selectors based on site structure)
    paragraphs = soup.find_all("p")
    content = " ".join([p.get_text() for p in paragraphs])

    data.append({"text": content})

# Save the scraped text data
with open("scraped_text_data.json", "w") as f:
    json.dump(data, f)

print("Scraped data saved successfully!")
