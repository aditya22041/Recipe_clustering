import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import json
import numpy as np

# Download 'punkt' tokenizer data if not already downloaded
nltk.download('punkt')

# Read the Excel file
df = pd.read_excel(r'ip\dish.xlsx') 

# Replace NaN with an empty string
df = df.fillna('')

# Prepare data for clustering
recipe_titles = df['Recipe_title'].tolist()
characteristics = df[['Taste', 'Odour', 'Colour', 'Texture', 'Description']].astype(str).values.tolist()

# Combine the characteristics into a single string for each recipe
combined_data = [" ".join(row) for row in characteristics]

# Step 1: TF-IDF Vectorization with more parameters
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',  # Removes common stop words
    token_pattern=r'\b\w+\b',  # Removes punctuation
    ngram_range=(1, 2),  # Unigrams + Bigrams
    min_df=2,  # Words must appear in at least 2 documents
    max_df=0.95  # Ignore words in more than 95% of documents
)
tfidf_matrix = vectorizer.fit_transform(combined_data)

# Step 2: Encode the ground truth labels into numerical values
ground_truth = df['Ground_Truth']
label_encoder = LabelEncoder()
ground_truth_encoded = label_encoder.fit_transform(ground_truth)

# Step 3: Loop over PCA components and calculate ARI scores with additional parameters
pca_values = np.arange(0.8, 0.96, 0.01)
ari_scores = []
best_ari = -1
best_pca_value = None

for pca_value in pca_values:
    # Apply PCA with additional parameters such as whiten=True for better performance on some datasets.
    pca = PCA(n_components=pca_value, whiten=True)
    tfidf_reduced = pca.fit_transform(tfidf_matrix.toarray())

    # Normalize the data
    tfidf_normalized = normalize(tfidf_reduced, norm='l2')

    # Apply K-Means Clustering
    num_clusters = 33  # Example: use at least 15 clusters
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(tfidf_normalized)

    # Compute the Adjusted Rand Index (ARI)
    ari_score = adjusted_rand_score(ground_truth_encoded, labels)
    
    # Update the best ARI score and corresponding PCA value if applicable
    if ari_score > best_ari:
        best_ari = ari_score
        best_pca_value = pca_value

# Step 4: Print the best ARI score and PCA value in specified format
best_parameters = {
    "VECTORIZATION": "TF-IDF",
    "DIMENSIONALITY": "PCA",
    "ARI": best_ari,
    "Parameters of vectorization method": {
        "max_features": vectorizer.max_features,
        "stop_words": vectorizer.stop_words,
        "ngram_range": vectorizer.ngram_range,
        "min_df": vectorizer.min_df,
        "max_df": vectorizer.max_df,
    },
    "Parameters Of Dimensionality Reduction": {
        "n_components": best_pca_value,
        "whiten": True,
    }
}

print(f"Best ARI Score: {best_ari:.3f}")
print(f"Best Parameters: {best_parameters}")

# Step 5: Save clustering results for the best PCA value to a file in specified format 
clustering_results = {
    "VECTORIZATION": "TF-IDF",
    "DIMENSIONALITY": "PCA",
    "ARI": best_ari,
    "Parameters of vectorization method": {
        "max_features": vectorizer.max_features,
        "stop_words": vectorizer.stop_words,
        "ngram_range": vectorizer.ngram_range,
        "min_df": vectorizer.min_df,
        "max_df": vectorizer.max_df,
    },
    "Parameters Of Dimensionality Reduction": {
        "n_components": best_pca_value,
        "whiten": True,
    }
}

with open('best_clustering_results_ifpca.json', 'w') as json_file:
    json.dump(clustering_results, json_file, indent=4)

print("Best clustering results saved to 'best_clustering_results_ifpca.json'.")
