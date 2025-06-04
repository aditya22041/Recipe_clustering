import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import json

# Download 'punkt' tokenizer data if not already downloaded
nltk.download('punkt')

# Read the Excel file
df = pd.read_excel('dish.xlsx')

# Prepare data for clustering
recipe_titles = df['Recipe_title'].tolist()
characteristics = df[['Taste', 'Odour', 'Colour', 'Texture', 'Description']].astype(str).values.tolist()

# Combine the characteristics into a single string for each recipe
combined_data = [" ".join(row) for row in characteristics]

# Tokenize the combined characteristics
tokenized_data = [" ".join(word_tokenize(doc.lower())) for doc in combined_data]

# Step 1: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(tokenized_data)

# Step 2: Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)
tfidf_reduced = pca.fit_transform(tfidf_matrix.toarray())

# Step 3: Normalize the data
tfidf_normalized = normalize(tfidf_reduced, norm='l2')

# Step 4: Apply DBSCAN Clustering with Parameter Tuning
best_eps = None
best_min_samples = None
best_score = -1

# Grid search over eps and min_samples
eps_values = np.linspace(0.1, 1.0, 10)
min_samples_values = range(2, 21)
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(tfidf_normalized)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if num_clusters > 1:  # More than one cluster
            try:
                score = silhouette_score(tfidf_normalized, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
            except ValueError:
                # Silhouette score cannot be computed for less than 2 clusters
                pass

print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best silhouette score: {best_score}")

# Use the best parameters found
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels = dbscan.fit_predict(tfidf_normalized)

# Separate noise points and non-noise points
noise_indices = [i for i, label in enumerate(labels) if label == -1]
non_noise_indices = [i for i, label in enumerate(labels) if label != -1]

noise_data = tfidf_normalized[noise_indices]
non_noise_labels = [labels[i] for i in non_noise_indices]

# Re-cluster noise points separately
if len(noise_indices) > 0:
    dbscan_noise = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    noise_labels = dbscan_noise.fit_predict(noise_data)
    # Map noise points to the nearest existing cluster
    noise_labels_adjusted = []
    for label in noise_labels:
        if label == -1:
            # If still noise, assign to the nearest cluster
            noise_labels_adjusted.append(min(non_noise_labels, key=lambda x: np.linalg.norm(tfidf_normalized[noise_indices[0]] - tfidf_normalized[non_noise_indices[x]])))
        else:
            noise_labels_adjusted.append(label)

    # Update labels for noise points
    for idx, noise_idx in enumerate(noise_indices):
        labels[noise_idx] = noise_labels_adjusted[idx]

# Step 5: Dimensionality Reduction for Visualization
pca_2d = PCA(n_components=2)
tfidf_2d = pca_2d.fit_transform(tfidf_normalized)

# Step 6: Plot Clusters with Adjusted Size
plt.figure(figsize=(14, 10))  # Increased the height from 6 to 10

# Subplot 1: Scatter Plot of Clusters
plt.subplot(2, 1, 1)  # Changed to 2 rows and 1 column
unique_labels = set(labels)
colors = [plt.cm.jet(float(i) / len(unique_labels)) for i in range(len(unique_labels))]
for cluster_id in unique_labels:
    if cluster_id == -1:
        color = 'k'
        label = 'Noise'
    else:
        color = colors[cluster_id]
        label = f'Cluster {cluster_id}'
    plt.scatter(
        tfidf_2d[labels == cluster_id, 0],
        tfidf_2d[labels == cluster_id, 1],
        c=color,
        label=label
    )
plt.title('DBSCAN Clustering of Recipes')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# Subplot 2: Bar Graph of Cluster Sizes with Numbers
plt.subplot(2, 1, 2)  # Changed to 2 rows and 1 column
cluster_sizes = [list(labels).count(cluster_id) for cluster_id in unique_labels]
cluster_labels = [f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise' for cluster_id in unique_labels]
bars = plt.bar(cluster_labels, cluster_sizes, color=[colors[i] for i in unique_labels])
plt.title('Number of Items in Each Cluster')
plt.xlabel('Clusters')
plt.ylabel('Number of Items')
plt.xticks(rotation=45, ha='right')

# Add number labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height}',
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()

# Step 7: Save Clustering Results to a File
clustering_results = {}
for cluster_id in unique_labels:
    if cluster_id != -1:  # Exclude original noise points
        clustering_results[f'Cluster {cluster_id}'] = [
            recipe_titles[i] for i in range(len(recipe_titles)) if labels[i] == cluster_id
        ]

with open('clustering_results.json', 'w') as json_file:
    json.dump(clustering_results, json_file, indent=4)

print("Clustering results saved to 'clustering_results.json'.")
