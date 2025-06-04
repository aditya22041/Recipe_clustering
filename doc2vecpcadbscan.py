import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor

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

# Step 1: Define parameters for Doc2Vec models (DBOW and DM)
doc2vec_params = [
    {'model_type': 'DBOW', 'vector_size': 100, 'window': 5, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DBOW', 'vector_size': 100, 'window': 6, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DBOW', 'vector_size': 100, 'window': 7, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DM', 'vector_size': 100, 'window': 5, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DM', 'vector_size': 100, 'window': 6, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DM', 'vector_size': 100, 'window': 7, 'min_count': 1, 'epochs': 40}
]

# Step 2: TF-IDF Vectorization with additional parameters
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    token_pattern=r'\b\w+\b',
    ngram_range=(1, 2), 
    min_df=2,
    max_df=0.95  
)

# Prepare to store results for both methods
results_summary = []

# Encode the ground truth labels into numerical values
if 'Ground_Truth' in df.columns:
    ground_truth = df['Cluster_Number']
else:
    print("Column 'Ground_truth' not found in the DataFrame.")
    # Add alternative actions here if needed

label_encoder = LabelEncoder()
ground_truth_encoded = label_encoder.fit_transform(ground_truth)

# Parameters for DBSCAN
eps_values = [0.2,0.3, 0.5,  0.9]
min_samples_values = [5, 6, 10,12, 15,20]

for params in doc2vec_params:
    # Step 3: Prepare TaggedDocuments for Doc2Vec
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(combined_data)]

    # Train the Doc2Vec model based on the type (DBOW or DM)
    if params['model_type'] == 'DBOW':
        model = Doc2Vec(vector_size=params['vector_size'],
                        window=params['window'],
                        min_count=params['min_count'],
                        workers=4,
                        epochs=params['epochs'],
                        dm=0)  # DBOW mode
    else:
        model = Doc2Vec(vector_size=params['vector_size'],
                        window=params['window'],
                        min_count=params['min_count'],
                        workers=4,
                        epochs=params['epochs'],
                        dm=1)  # DM mode

    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Create document vectors using the trained model
    doc_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in combined_data]



    # Step 4: Loop over PCA components and calculate ARI scores
    pca_values = np.arange(0.8, 0.96, 0.01)
    
    for pca_value in pca_values:
        pca = PCA(n_components=pca_value, whiten=True)
        doc_vectors_reduced = pca.fit_transform(doc_vectors)

        # Normalize the data
        doc_vectors_normalized = normalize(doc_vectors_reduced, norm='l2')

        # Test DBSCAN with different eps and min_samples
        for eps in eps_values:
            for min_samples in min_samples_values:
                # Apply DBSCAN Clustering
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(doc_vectors_normalized)

                # Filter out noise labels (-1) if any clusters exist
                if len(set(labels)) > 1:
                    ari_score = adjusted_rand_score(ground_truth_encoded[labels != -1], labels[labels != -1])
                else:
                    ari_score = -1

                # Save results in structured format
                results_summary.append({
                    "VECTORIZATION": params['model_type'],
                    "DIMENSIONALITY": "PCA",
                    "ARI": ari_score,
                    "Parameters of vectorization method": {
                        "model_type": params['model_type'],
                        "vector_size": params['vector_size'],
                        "window": params['window'],
                        "min_count": params['min_count'],
                        "epochs": params['epochs']
                    },
                    "Parameters Of Dimensionality Reduction": {
                        "n_components": pca_value,
                        "whiten": True,
                    },
                    "DBSCAN Parameters": {
                        "eps": eps,
                        "min_samples": min_samples
                    }
                })

# Find the best ARI score and parameters
best_result = max(results_summary, key=lambda x: x["ARI"])
print(f"Best ARI Score: {best_result['ARI']:.3f}")
print(f"Best Parameters: {best_result}")

# Save results to JSON file
with open('best_clustering_results_combined.json', 'w') as json_file:
    json.dump(results_summary, json_file, indent=4)

print("Best clustering results saved to 'best_clustering_results_combined.json'.")
