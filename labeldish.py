import pandas as pd
import requests
import string
from bs4 import BeautifulSoup
import re

file_path = "recipe.xlsx"
taste_set = set(['sweet', 'sour', 'salty', 'bitter', 'umami'])
texture_list = set(["creamy", "smooth", "rough", "gritty", "fatty", "juicy", "tender", "chewy", "crunchy", "crispy", "soggy", "stringy", "sticky", "flaky", "grainy", "greasy", "oily", "dry", "moist", "soft", "hard", "tough", "rubbery", "spongy", "succulent", "watery", "fizzy", "sparkling", "flat", "bubbly", "effervescent", "carbonated", "still", "silky", "velvety", "syrupy", "viscous"])
aroma_list = set(["floral", "fruity", "herbal", "nutty", "spicy", "smoky", "woody", "grassy", "earthy", "minty", "pungent", "rancid", "stale", "fresh"])
flavor_terms = set(["astringent", "acrid", "buttery", "citrusy", "earthy", "floral", "fruity", "herbal", "nutty", "smoky", "spicy", "zesty", "bland", "bitter", "sour", "sweet", "salty", "umami", "savory", "sharp", "tangy", "tart", "mild", "peppery", "pungent", "rich", "robust", "strong", "subtle", "tasteless", "tasty", "delicious", "yummy", "appetizing", "flavorful", "palatable", "delectable", "toothsome", "scrumptious", "mouthwatering", "luscious", "lip-smacking", "nectarous", "sapid", "succulent", "tangy", "zesty", "piquant", "spicy", "fiery", "peppery", "pungent", "hot", "bland", "mild", "tasteless", "unseasoned", "unflavored", "unsavory", "unappetizing", "unpalatable", "unsatisfying", "unpleasant", "disgusting", "nauseating", "repulsive", "offensive", "revolting", "vile", "foul", "putrid", "rancid", "rank", "spoiled", "decayed", "moldy", "musty", "stale", "rotten", "bad", "sour", "bitter", "salty", "sweet", "savory", "umami", "tangy", "tart", "acidic", "sourish", "soury", "sour-tasting", "sour-flavored", "acidic-tasting", "acidic-flavored", "acidulous", "acetic", "vinegary", "sharp", "biting", "tart-tasting", "tart-flavored", "tartish", "sharpish", "pungentish", "bitingish", "astringentish", "sour-tasting", "sour-flavored"])

# Read the Excel file
df = pd.read_excel(file_path)  # Update sheet name if needed
dish_names = df['Recipe_title'].tolist()
COUNT=0
# Ensure the necessary columns are present and set their dtype to str
for column in ['Taste', 'Texture', 'Aroma', 'Flavor', 'Temperature','wiki']:
    if column not in df.columns:
        df[column] = ""
    df[column] = df[column].astype(str)

def clean_keyword(keyword):
    # Remove text within parentheses
    keyword = re.sub(r'\(.*?\)', '', keyword)
    # Split by conjunctions like 'or'
    keyword_parts = re.split(r'\s+or\s+|\s+and\s+', keyword)
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
    except requests.exceptions.HTTPError:
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return None

def extract_information(soup, index, df):
    # Extract infobox details
    infobox = soup.find('table', {'class': 'infobox'})
    if infobox:
        rows = infobox.find_all('tr')
        for row in rows:
            heading = row.find('th')
            detail = row.find('td')
            if heading and detail:
                print(f"{heading.text.strip()}  ::  {detail.text.strip()}")
                if heading.text.strip() == "Serving temperature":
                    print("Temperature: ", detail.text.strip())
                    df.at[index, 'Temperature'] = detail.text.strip()

def wikibot(keyword, index, df,COUNT):
    
    # Clean and split the keyword
    keyword_variants = clean_keyword(keyword)

    for variant in keyword_variants:
        content = fetch_wikipedia_page(variant)
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            extract_information(soup, index, df)
            break
    else:
        print(f"Error: Could not fetch the Wikipedia page for any variants of {keyword}")

    if content:
        soup = BeautifulSoup(content, 'html.parser')
        paragraphs = soup.find_all('p')
        print("Paragraphs: ", len(paragraphs))

        tasteset = set()
        texturelist = set()
        aromalist = set()
        flavorlist = set()

        for paragraph in paragraphs:
            text_content = paragraph.text.strip()
            for word in text_content.split():
                word_lower = word.lower()
                if word_lower in taste_set:
                    tasteset.add(word_lower)
                if word_lower in texture_list:
                    texturelist.add(word_lower)
                if word_lower in aroma_list:
                    aromalist.add(word_lower)
                if word_lower in flavor_terms:
                    flavorlist.add(word_lower)

        # Print the found categories
        print("Taste: ", tasteset)
        print("Texture: ", texturelist)
        print("Aroma: ", aromalist)
        print("Flavor: ", flavorlist)

        # Join the categories into comma-separated strings
        df.at[index, 'Taste'] = ", ".join(tasteset)
        df.at[index, 'Texture'] = ", ".join(texturelist)
        df.at[index, 'Aroma'] = ", ".join(aromalist)
        df.at[index, 'Flavor'] = ", ".join(flavorlist)

        # Save the updated DataFrame back to the same Excel file
        df.to_excel(file_path, index=False)
        COUNT+=1
    else:
        print(f"Error: Could not fetch the Wikipedia page for {keyword}")
        df.at[index, 'wiki'] ="no wiki"
        df.to_excel(file_path, index=False)


    return df

# Run for  whole and write the data to a file
for i, dish_name in enumerate(dish_names):
    print(f"\nFetching data for: {dish_name}")
    print("=" * 50)
    wikibot(dish_name, i, df,COUNT)
    print("=" * 50)
print(COUNT)
