import requests
import re
import string
from bs4 import BeautifulSoup
import nltk
import json
import pandas as pd

# Download 'punkt' tokenizer data if not already downloaded
nltk.download('punkt')

# Define helper functions
def clean_keyword(keyword):
    keyword = re.sub(r'\(.*?\)', '', str(keyword))
    keyword_parts = re.split(r'\s+or\s+|\s+and\s+', str(keyword))
    return [part.strip() for part in keyword_parts]

def fetch_wikipedia_page(keyword):
    u_i = string.capwords(keyword)
    lists = u_i.split()
    word = "_".join(lists)
    url = "https://en.wikipedia.org/wiki/" + word
    try:
        url_open = requests.get(url)
        url_open.raise_for_status()
        return url_open.content
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return None

def extract_information(soup):
    summary = ""
    temperature = ""
    paragraphs = soup.find_all('p')
    if paragraphs:
        summary = ' '.join([para.text.strip() for para in paragraphs])
    infobox = soup.find('table', {'class': 'infobox'})
    if infobox:
        rows = infobox.find_all('tr')
        for row in rows:
            heading = row.find('th')
            detail = row.find('td')
            if heading and detail and heading.text.strip() == "Serving temperature":
                temperature = detail.text.strip()
                break
    if temperature:
        summary += " " + temperature
    return summary

def wikibot(keyword):
    keyword_variants = clean_keyword(keyword)
    content = None
    for variant in keyword_variants:
        content = fetch_wikipedia_page(variant)
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            return extract_information(soup)
    else:
        print(f"Error: Could not fetch the Wikipedia page for any variants of {keyword}")
        return ""

# Read recipe names from the Excel sheet
excel_file = 'dish.xlsx'
df = pd.read_excel(excel_file)

# Ensure the column name is correct
if 'Recipe_title' not in df.columns:
    raise ValueError("Column 'Recipe_title' not found in the Excel file.")

# Initialize summary column
summaries = {}

# Fetch Wikipedia summaries for recipes
for index, row in df.iterrows():
    summaries[row['Recipe_title']] = wikibot(row['Recipe_title'])

# Save the summaries to a JSON file
with open('dish_summaries.json', 'w') as json_file:
    json.dump(summaries, json_file)

print("Data fetched and saved successfully.")
