import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np
import json
import umap
import time
from concurrent.futures import ProcessPoolExecutor
import warnings
from scipy.optimize import OptimizeWarning  # Import OptimizeWarning

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

    # Step 1: Train the Doc2Vec model # 10  window  # another doc
    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Step 2: Create document vectors using the trained model
    doc_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in descriptions]

    # Step 3: Encode the ground truth labels into numerical values
    ground_truth = df['Ground_Truth']
    label_encoder = LabelEncoder()
    ground_truth_encoded = label_encoder.fit_transform(ground_truth)

    # Step 4: Optimize UMAP parameters using parallel processing
    ari_scores = []
    best_ari = -1
    best_labels = None

    # Collecting data for plotting ARI scores against parameters
    score_data = []

    n_neighbors_options = [5, 10]
    min_dist_options = [0.0, 0.1]
    n_components_options = [2]   # 2,3.
    metric_options = ['euclidean', 'manhattan', 'cosine']
    spread_options = [0.1, 1.0]  

    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = []
        for n_neighbors in n_neighbors_options:
            for min_dist in min_dist_options:
                for n_components in n_components_options:
                    for metric in metric_options:
                        for spread in spread_options:
                            futures.append(executor.submit(cluster_with_umap,
                                                            doc_vectors,
                                                            ground_truth_encoded,
                                                            n_neighbors,
                                                            min_dist,
                                                            n_components,
                                                            metric,
                                                            spread))

        for future in futures:
            n_neighbors, min_dist, n_components, metric, spread, ari_score, labels = future.result()
            ari_scores.append((n_neighbors, min_dist, n_components, metric, spread, ari_score))

            # Collect score data for plotting ARI scores against parameters
            score_data.append((n_neighbors, min_dist, ari_score))
            
            # Update the best ARI score if applicable
            if ari_score > best_ari:
                best_ari = ari_score
                best_labels = labels

    # Plotting ARI scores against parameters
    scores_array = np.array(score_data)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(scores_array[:, 0], scores_array[:, 1], c=scores_array[:, 2], cmap='viridis', s=100)
    plt.colorbar(scatter, label='ARI Score')
    plt.xlabel('n_neighbors')
    plt.ylabel('min_dist')
    plt.title('ARI Score Changes with UMAP Parameters')
    plt.grid(True)
    plt.show()

    # Step 5: Normalize document vectors using the best UMAP parameters found
    if best_labels is not None:
        best_params = [(n_neighbors, min_dist, n_components, metric, spread) 
                       for n_neighbors, min_dist, n_components, metric, spread, score in ari_scores if score == best_ari]
        n_neighbors_best, min_dist_best, n_components_best, metric_best, spread_best = best_params[0]

        umap_model_best = umap.UMAP(n_neighbors=n_neighbors_best,
                                     min_dist=min_dist_best,
                                     n_components=n_components_best,
                                     metric=metric_best,
                                     spread=spread_best)
        doc_vectors_reduced_best = umap_model_best.fit_transform(doc_vectors)
        doc_vectors_normalized = normalize(doc_vectors_reduced_best, norm='l2')

        num_clusters_best = len(set(ground_truth_encoded))
        kmeans_best = MiniBatchKMeans(n_clusters=num_clusters_best, random_state=42)
        final_labels = kmeans_best.fit_predict(doc_vectors_normalized)

        plt.figure(figsize=(8, 6))
        plt.scatter(doc_vectors_normalized[:, 0], doc_vectors_normalized[:, 1], c=final_labels, cmap='Spectral', s=50)
        plt.title('UMAP Projection of Document Vectors')
        plt.colorbar(label='Cluster ID')
        plt.show()

        print(f"Best ARI Score: {best_ari:.3f}")

        if best_labels is not None:
            print(f"Best Parameters: {best_params}")

        cluster_centers_best = kmeans_best.cluster_centers_

        key_characteristics = []
        for center in cluster_centers_best:
            top_indices = np.argsort(center)[::-1][:5]  
            top_features = [df['Recipe_title'].iloc[i] for i in top_indices]
            key_characteristics.append(", ".join(top_features))

        clustering_results = {}
        unique_labels_best = set(final_labels)

        for cluster_id in unique_labels_best:
            clustering_results[f'Cluster {cluster_id}'] = {
                'Characteristics': key_characteristics[cluster_id],
                'Recipes': [
                    df['Recipe_title'].iloc[i] for i in range(len(df)) if final_labels[i] == cluster_id
                ]
            }

        with open('best_clustering_results.json', 'w') as json_file:
            json.dump(clustering_results, json_file, indent=4)

        print("Best clustering results saved to 'best_clustering_results.json'.")

if __name__ == '__main__':
   main()