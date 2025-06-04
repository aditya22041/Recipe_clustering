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
from sklearn.cluster import DBSCAN

# Download 'punkt' tokenizer data if not already downloaded
nltk.download('punkt')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

def cluster_with_umap(doc_vectors, ground_truth_encoded, eps, min_samples, metric, algorithm, leaf_size):
    umap_model = umap.UMAP(n_neighbors=5,
                           min_dist=0.0,
                           n_components=2,
                           metric='euclidean',
                           spread=0.5)
    
    doc_vectors_reduced = umap_model.fit_transform(doc_vectors)
    doc_vectors_normalized = normalize(doc_vectors_reduced, norm='l2')
    
    # Use DBSCAN with additional parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size)
    
    labels = dbscan.fit_predict(doc_vectors_normalized)
    
    if len(set(labels)) > 1:
        ari_score = adjusted_rand_score(ground_truth_encoded[labels != -1], labels[labels != -1])
    else:
        ari_score = -1
    
    return (ari_score, labels)

def main():
    # Read the Excel file for ground truth and recipe titles
    df = pd.read_excel(r'ip\dish.xlsx')  # Use raw string for the path

    # Replace NaN with an empty string and limit data size for optimization
    df = df.fillna('')
    df = df.sample(frac=0.5)  # Use only 50% of the data for faster processing

    # Combine relevant columns into a single description for each recipe
    df['Combined_Description'] = df['Taste'] + ' ' + df['Odour'] + ' ' + df['Colour'] + ' ' + df['Texture'] + ' ' + df['Description']

    # Prepare data for clustering using combined descriptions
    descriptions = df['Combined_Description'].tolist()

    # Tokenize the descriptions for Doc2Vec
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(descriptions)]

    doc2vec_params = [
        {'model_type': 'DBOW', 'vector_size': 100, 'window': 8, 'min_count': 1, 'epochs': 40},
        {'model_type': 'DBOW', 'vector_size': 100, 'window': 6, 'min_count': 1, 'epochs': 40},
        {'model_type': 'DBOW', 'vector_size': 100, 'window': 7, 'min_count': 1, 'epochs': 40},
        {'model_type': 'DM', 'vector_size': 100, 'window': 8, 'min_count': 1, 'epochs': 40},
        {'model_type': 'DM', 'vector_size': 100, 'window': 6, 'min_count': 1, 'epochs': 40},
        {'model_type': 'DM', 'vector_size': 100, 'window': 7, 'min_count': 1, 'epochs': 40}
    ]

    best_results = []
    
    eps_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  #
    min_samples_options = [2, 3, 5, 10, 15]
    algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size_options = [10, 30, 50]

    # Define valid metrics for each algorithm to avoid incompatible combinations
    valid_metrics = {
        'auto': ['euclidean', 'manhattan', 'cosine'],
        'ball_tree': ['euclidean', 'manhattan'],
        'kd_tree': ['euclidean', 'manhattan'],
        'brute': ['euclidean', 'manhattan', 'cosine']
    }

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
            model = Doc2Vec(vector_size=vector_size,
                            window=window,
                            min_count=min_count,
                            workers=4,
                            epochs=epochs,
                            dm=(0 if model_type == 'DBOW' else 1))  # DBOW or DM mode

            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

            doc_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in descriptions]

            for eps in eps_options:
                for min_samples in min_samples_options:
                    for algorithm in algorithm_options:
                        for metric in valid_metrics[algorithm]:
                            for leaf_size in leaf_size_options:
                                futures.append(executor.submit(cluster_with_umap,
                                                              doc_vectors,
                                                              ground_truth_encoded,
                                                              eps,
                                                              min_samples,
                                                              metric,
                                                              algorithm,
                                                              leaf_size))

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
                        "epochs": epochs
                    },
                    "DBSCAN Parameters": {
                        "eps": eps,
                        "min_samples": min_samples,
                        "metric": metric,
                        "algorithm": algorithm,
                        "leaf_size": leaf_size
                    }
                })

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if elapsed_time > 600:  # Check if execution time exceeds 10 minutes (600 seconds)
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