import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import json
import numpy as np
import umap
import time
from concurrent.futures import ProcessPoolExecutor
import warnings
from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    return (n_neighbors, min_dist, n_components, metric, spread, ari_score, labels, doc_vectors_reduced)

def main():
    # Load the dataset
    df = pd.read_excel(r'ip\dish.xlsx').fillna('')
    df['Combined_Description'] = df['Taste'] + ' ' + df['Odour'] + ' ' + df['Colour'] + ' ' + df['Texture'] + ' ' + df['Description']
    descriptions = df['Combined_Description'].tolist()

    # Prepare tagged data for Doc2Vec
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(descriptions)]
    
    # Parameters for both DM and DBOW models
    doc2vec_params = [
        {'model_type': 'DM', 'vector_size': 100, 'window': 5, 'min_count': 1, 'epochs': 40, 'dm': 1},
        {'model_type': 'DBOW', 'vector_size': 100, 'window': 5, 'min_count': 1, 'epochs': 40, 'dm': 0}
    ]
    
    best_results = []
    n_neighbors_options = [9, 10,11]
    min_dist_options = [0.1, 0.12]
    spread_options = [0.2, 0.3, 0.5]
    n_components_options = [2]
    metric_options = ['euclidean', 'manhattan']

    # Encode labels
    label_encoder = LabelEncoder()
    ground_truth_encoded = label_encoder.fit_transform(df['Ground_Truth'])

    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = []
        
        for params in doc2vec_params:
            model_type, vector_size, window, min_count, epochs, dm = params.values()
            model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=4, epochs=epochs, dm=dm)
            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

            doc_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in descriptions]

            for n_neighbors in n_neighbors_options:
                for min_dist in min_dist_options:
                    for spread in spread_options:
                        if min_dist <= spread:
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
                    "MODEL_TYPE": params['model_type'],
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
                    },
                    "labels": result[6].tolist() if isinstance(result[6], np.ndarray) else result[6],
                    "doc_vectors_reduced": result[7].tolist() if isinstance(result[7], np.ndarray) else result[7],
                })

    best_result = max(best_results, key=lambda x: x["ARI"])
    results_summary = {
        "best_results": best_results,
        "best_ari_score": best_result["ARI"],
        "best_parameters": {
            "vectorization_params": best_result["Parameters of vectorization method"],
            "dimensionality_reduction_params": best_result["Parameters Of Dimensionality Reduction"]
        }
    }

    with open('best_clustering_results.json', 'w') as json_file:
        json.dump(results_summary, json_file, indent=4)

    print(f"Best ARI Score: {results_summary['best_ari_score']:.3f}")
    print("Best parameters for best ARI score:")
    print("Vectorization parameters:", results_summary["best_parameters"]["vectorization_params"])
    print("Dimensionality reduction parameters:", results_summary["best_parameters"]["dimensionality_reduction_params"])

    # Visualization for best result
    doc_vectors_reduced = np.array(best_result['doc_vectors_reduced'])
    labels = best_result['labels']
    ground_truth_labels = df['Ground_Truth']

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=doc_vectors_reduced[:, 0], y=doc_vectors_reduced[:, 1], hue=labels, palette='viridis', style=ground_truth_labels)
    plt.title(f"UMAP Visualization with KMeans Clustering (Best ARI: {best_result['ARI']:.3f}) - {best_result['MODEL_TYPE']} Model")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
