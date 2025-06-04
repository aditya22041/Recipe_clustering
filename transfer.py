import pandas as pd
import requests

def wikipedia_page_exists(title):
    url = "https://en.wikipedia.org/wiki/" + str(title)
    response = requests.get(url)
    return response.status_code == 200

# Step 1: Read the first column from recipe.xlsx starting from row 501 to 1000
recipe_df = pd.read_excel('recipe.xlsx', usecols=[0], skiprows=range(1, 3760), nrows=200)
recipe_titles = recipe_df.iloc[:, 0].tolist()

# Load existing dish.xlsx or create a new DataFrame if it doesn't exist
try:
    dish_df = pd.read_excel('dish.xlsx')
except FileNotFoundError:
    dish_df = pd.DataFrame(columns=['Recipe_title'])

# Step 2 and 3: Check if Wikipedia page exists for each title and append to dish_df if it does
for title in recipe_titles:
    if wikipedia_page_exists(title):
        print(title)
        dish_df = dish_df._append({'Recipe_title': title}, ignore_index=True)

# Save the updated DataFrame back to dish.xlsx
with pd.ExcelWriter('dish.xlsx', engine='openpyxl') as writer:
    dish_df.to_excel(writer, index=False)

print("Process completed. Valid titles have been added to dish.xlsx.")
