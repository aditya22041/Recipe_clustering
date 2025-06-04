import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import adjusted_rand_score
import json
import umap
import time
from concurrent.futures import ProcessPoolExecutor
import warnings
from scipy.optimize import OptimizeWarning
import hdbscan

# Download 'punkt' tokenizer data if not already downloaded
nltk.download('punkt')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

def cluster_with_umap(doc_vectors, ground_truth_encoded, min_cluster_size, min_samples, umap_params, hdbscan_params):
    umap_model = umap.UMAP(**umap_params)
    
    doc_vectors_reduced = umap_model.fit_transform(doc_vectors)
    doc_vectors_normalized = normalize(doc_vectors_reduced, norm='l2')
    
    # Use HDBSCAN with additional parameters
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, **hdbscan_params)
    labels = hdbscan_clusterer.fit_predict(doc_vectors_normalized)
    
    if len(set(labels)) > 1:
        ari_score = adjusted_rand_score(ground_truth_encoded[labels != -1], labels[labels != -1])
    else:
        ari_score = -1
    
    return (ari_score, labels)

def main():
    df = pd.read_excel(r'ip\dish.xlsx')
    df.columns = df.columns.str.strip()
    
    if 'Cluster_Number' not in df.columns:
        raise ValueError("Column 'Cluster_Number' not found in DataFrame.")

    df = df.fillna('')
    df = df.sample(frac=0.5)
    df['Combined_Description'] = df['Taste'] + ' ' + df['Odour'] + ' ' + df['Colour'] + ' ' + df['Texture'] + ' ' + df['Description']
    descriptions = df['Combined_Description'].tolist()

    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(descriptions)]

    doc2vec_params = [
        {'model_type': 'DBOW', 'vector_size': 100, 'window': 8, 'min_count': 1, 'epochs': 40},
        {'model_type': 'DM', 'vector_size': 50, 'window': 5, 'min_count': 1, 'epochs': 40, 'dm_mean': 0},
        {'model_type': 'DM', 'vector_size': 150, 'window': 10, 'min_count': 1, 'epochs': 30, 'dm_mean': 1}
    ]

    umap_params_options = [
        {'n_neighbors': 5, 'min_dist': 0.0, 'n_components': 2, 'metric': 'euclidean', 'spread': 0.5},
        {'n_neighbors': 10, 'min_dist': 0.1, 'n_components': 2, 'metric': 'manhattan', 'spread': 1.0}
    ]

    hdbscan_params_options = [
        {'cluster_selection_method': 'eom', 'metric': 'euclidean'},
        {'cluster_selection_method': 'leaf', 'metric': 'manhattan'}
    ]

    min_cluster_size_options = [5, 10, 15, 20]#  11 12 13 14 16 17 18 19
    min_samples_options = [1, 5, 10,15,20]

    ground_truth = df['Cluster_Number']
    label_encoder = LabelEncoder()
    ground_truth_encoded = label_encoder.fit_transform(ground_truth)

    start_time = time.time()
    best_results = []

    with ProcessPoolExecutor() as executor:
        futures = []

        for params in doc2vec_params:
            model_type = params['model_type']
            vector_size = params['vector_size']
            window = params['window']
            min_count = params['min_count']
            epochs = params['epochs']
            dm_mean = params.get('dm_mean', 0)

            model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=4, epochs=epochs, dm=(0 if model_type == 'DBOW' else 1), dm_mean=dm_mean)
            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
            doc_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in descriptions]

            for umap_params in umap_params_options:
                for hdbscan_params in hdbscan_params_options:
                    for min_cluster_size in min_cluster_size_options:
                        for min_samples in min_samples_options:
                            futures.append(executor.submit(cluster_with_umap, doc_vectors, ground_truth_encoded, min_cluster_size, min_samples, umap_params, hdbscan_params))

        for future in futures:
            result = future.result()
            if result is not None:
                best_results.append({
                    "VECTORIZATION": "Doc2Vec",
                    "DIMENSIONALITY": "UMAP",
                    "ARI": result[0],
                    "Parameters of vectorization method": {
                        "model_type": model_type,
                        "vector_size": vector_size,
                        "window": window,
                        "min_count": min_count,
                        "epochs": epochs,
                        "dm_mean": dm_mean
                    },
                    "UMAP Parameters": umap_params,
                    "HDBSCAN Parameters": {
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        **hdbscan_params
                    }
                })

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if elapsed_time > 600:
            print("Execution time exceeded 10 minutes.")
            return

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
