import pandas as pd
import wikipedia
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.preprocessing import normalize
import openpyxl


def clean_keyword(keyword):
    """
    Cleans a keyword by removing text within parentheses and splitting by conjunctions.

    Args:
        keyword: The keyword to clean.

    Returns:
        A list of cleaned keyword parts.
    """
    # Remove text within parentheses
    keyword = re.sub(r'\(.*?\)', '', keyword)
    # Split by conjunctions like 'or' or 'and'
    keyword_parts = re.split(r'\s+or\s+|\s+and\s+', keyword)
    return [part.strip() for part in keyword_parts]

def download_sample_wikipage(keyword):
    """
    Fetches a Wikipedia page using the wikipedia library.

    Args:
        keyword: The keyword to search for on Wikipedia.

    Returns:
        The Wikipedia page content as a string.
    """
    try:
        page = wikipedia.page(keyword)
        return page.content
    except wikipedia.exceptions.PageError:
        print(f"Error: Page not found for keyword: {keyword}")
        return None

# def preprocess_text(text):
#     """
#     Preprocesses a text string for doc2vec.

#     Args:
#         text: The text string to preprocess.

#     Returns:
#         A list of tokens representing the preprocessed text.
#     """
#     # Lowercase the text
#     text = text.lower()

#     # Split the text into words (tokens)
#     tokens = text.split()

#     # Remove stop words and punctuation
#     stop_words = set(nltk.corpus.stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

#     return tokens

file_path = "recipe.xlsx"

# Read the Excel file
# df = pd.read_excel(file_path)  # Update sheet name if needed
# dish_names = df['Recipe_title'].tolist()
# tokenized_data =[]
# # Download and preprocess the pages
# documents = []
# for i, dish_name in enumerate(dish_names):
#     print(f"\nFetching data for: {dish_name}")
#     keywords = clean_keyword(dish_name)
#     for keyword in keywords:
#         text = download_sample_wikipage(keyword)
#         if text:
#            print(text)
#         else :
#             print("no wiki")


# # Train the doc2vec model
# model = Doc2Vec(documents, vector_size=100, window=5, min_count=2, workers=4, epochs=10)

# # Save the model
# model.save("wikipedia_doc2vec.model")

# # Example usage
# example_keyword = "gulab jamun"
# example_text = download_sample_wikipage(example_keyword)
# if example_text:
#     example_tokens = preprocess_text(example_text)
#     vector = model.infer_vector(example_tokens)
#     print(vector)
# data = [
#     "This is the first document",
#     "This is the second document",
#     "This is the third document",
#     "This is the fourth document"
# ]

# Tokenize the documents
# tokenized_data = [word_tokenize(doc.lower()) for doc in data]

# Step 1: Build Vocabulary
# word_counts = defaultdict(int)
# for doc in tokenized_data:
#     for word in doc:
#         word_counts[word] += 1

# vocab = sorted(word_counts.keys())

# word_to_idx = {word: idx for idx, word in enumerate(vocab)}
# idx_to_word = {idx: word for word, idx in word_to_idx.items()}
# vocab_size = len(vocab)

# # Step 2: Prepare Training Data
# tagged_data = []
# for i, doc in enumerate(tokenized_data):
#     tagged_data.append((doc, [f"DOC_{i}"]))

# # Step 3: Initialize Model Parameters
# vector_size = 20  # Dimensionality of the document vectors
# learning_rate = 0.01
# epochs = 50

# # Initialize document and word vectors randomly
# doc_vectors = np.random.rand(len(tagged_data), vector_size)
# word_vectors = np.random.rand(vocab_size, vector_size)

# # Step 4: Train Doc2Vec Model
# for epoch in range(epochs):
#     for doc, tags in tagged_data:
#         doc_vec_sum = np.zeros(vector_size)
#         for word in doc:
#             word_idx = word_to_idx[word]
#             doc_vec_sum += word_vectors[word_idx]

#         doc_idx = int(tags[0].split('_')[1])
#         doc_vectors[doc_idx] += learning_rate * doc_vec_sum / len(doc)

# # Step 5: Normalize Document Vectors
# doc_vectors_normalized = normalize(doc_vectors, norm='l2')

# # Step 6: Print Document Vectors
# for i, doc in enumerate(data):
#     print("Document", i+1, ":", doc)
#     print("Vector:", doc_vectors_normalized[i])
#     print()