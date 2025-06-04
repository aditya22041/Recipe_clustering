import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import json
import numpy as npwh
import umap
import time
from concurrent.futures import ProcessPoolExecutor
import warnings
from scipy.optimize import OptimizeWarning

# Download 'punkt' tokenizer data if not already downloaded
nltk.download('punkt')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

def cluster_with_umap(doc_vectors, ground_truth_encoded, n_neighbors, min_dist, n_components, metric, spread):
    umap_model = umap.UMAP(n_neighbors=n_neighbors,
                           min_dist=min_dist,
                           n_components=n_components,
                           metric=metric,
                           spread=spread)
    
    doc_vectors_reduced = umap_model.fit_transform(doc_vectors)
    doc_vectors_normalized = normalize(doc_vectors_reduced, norm='l2')
    
    num_clusters = len(set(ground_truth_encoded))
    
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    
    labels = kmeans.fit_predict(doc_vectors_normalized)
    
    ari_score = adjusted_rand_score(ground_truth_encoded, labels)
    
    return (n_neighbors, min_dist, n_components, metric, spread, ari_score, labels)

def main():
    # Read the Excel file for ground truth and recipe titles
    df = pd.read_excel(r'ip\dish.xlsx')  # Use raw string for the path

    # Replace NaN with an empty string
    df = df.fillna('')

    # Combine relevant columns into a single description for each recipe
    df['Combined_Description'] = df['Taste'] + ' ' + df['Odour'] + ' ' + df['Colour'] + ' ' + df['Texture'] + ' ' + df['Description']

    # Prepare data for clustering using combined descriptions
    descriptions = df['Combined_Description'].tolist()

    # Tokenize the descriptions for Doc2Vec
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(descriptions)]

    # Step 1: Define parameters for both types of Doc2Vec models (DBOW and DM)
    doc2vec_params = [
    {'model_type': 'DBOW', 'vector_size': 100, 'window': 5, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DBOW', 'vector_size': 100, 'window': 6, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DBOW', 'vector_size': 100, 'window': 7, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DM', 'vector_size': 100, 'window': 5, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DM', 'vector_size': 100, 'window': 6, 'min_count': 1, 'epochs': 40},
    {'model_type': 'DM', 'vector_size': 100, 'window': 7, 'min_count': 1, 'epochs': 40}
]

# negative sampling

    best_results = []
    
    # Expanded parameter options with constraints on min_dist and spread
    n_neighbors_options = [5,10,11,12,13,15]    # 2...10    
    min_dist_options = [0.0, 0.1,0.11,0.12,0.13]   #0.1..1.15         
    spread_options = [0.1,0.2,0.5,0.7,1.0]               # Ensure spread is >= min_dist
    n_components_options = [2, 3] # explore 2..3               
    metric_options = ['euclidean', 'manhattan', 'cosine']

    # Encode ground truth labels into numerical values
    ground_truth = df['Ground_Truth']  
    label_encoder = LabelEncoder()
    ground_truth_encoded = label_encoder.fit_transform(ground_truth)

    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = []
        
        for params in doc2vec_params:
            model_type = params['model_type']
            vector_size = params['vector_size']
            window = params['window']
            min_count = params['min_count']
            epochs = params['epochs']

            # Train the Doc2Vec model based on the type (DBOW or DM)
            if model_type == 'DBOW':
                model = Doc2Vec(vector_size=vector_size,
                                window=window,
                                min_count=min_count,
                                workers=4,
                                epochs=epochs,
                                dm=0)  # DBOW mode
            else:
                model = Doc2Vec(vector_size=vector_size,
                                window=window,
                                min_count=min_count,
                                workers=4,
                                epochs=epochs,
                                dm=1)  # DM mode

            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

            doc_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in descriptions]

            for n_neighbors in n_neighbors_options:
                for min_dist in min_dist_options:
                    for spread in spread_options:
                        if min_dist <= spread:  # Ensure this condition is met.
                            for n_components in n_components_options:
                                for metric in metric_options:
                                    futures.append(executor.submit(cluster_with_umap,
                                                                    doc_vectors,
                                                                    ground_truth_encoded,
                                                                    n_neighbors,
                                                                    min_dist,
                                                                    n_components,
                                                                    metric,
                                                                    spread))

        for future in futures:
            result = future.result()
            if result is not None:
                best_results.append({
                    "VECTORIZATION": "Doc2Vec",
                    "DIMENSIONALITY": "UMAP",
                    "ARI": result[5],
                    "Parameters of vectorization method": {
                        "model_type": params['model_type'],
                        "vector_size": vector_size,
                        "window": window,
                        "min_count": min_count,
                        "epochs": epochs
                    },
                    "Parameters Of Dimensionality Reduction": {
                        "n_neighbors": result[0],
                        "min_dist": result[1],
                        "n_components": result[2],
                        "metric": result[3],
                        "spread": result[4]
                    }
                })

    # Save results to JSON files including best ARI score and parameters.
    results_summary = {
        "best_results": best_results,
        "best_ari_score": max(best_results, key=lambda x: x["ARI"])["ARI"],
        "best_parameters": max(best_results, key=lambda x: x["ARI"]),
    }

    with open('best_clustering_results.json', 'w') as json_file:
        json.dump(results_summary, json_file, indent=4)

    print(f"Best ARI Score: {results_summary['best_ari_score']:.3f}")
    print(f"Best Parameters: {results_summary['best_parameters']}")
    print("Clustering results saved to 'best_clustering_results.json'.")

if __name__ == '__main__':
   main()