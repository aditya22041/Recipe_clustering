import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.preprocessing import normalize

# Sample data
data = [
    "This is the first document",
    "This is the second document",
    "This is the third document",
    "This is the fourth document"
]

# Tokenize the documents
tokenized_data = [word_tokenize(doc.lower()) for doc in data]

# Step 1: Build Vocabulary
word_counts = defaultdict(int)
for doc in tokenized_data:
    for word in doc:
        word_counts[word] += 1

vocab = sorted(word_counts.keys())

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)

# Step 2: Prepare Training Data
tagged_data = []
for i, doc in enumerate(tokenized_data):
    tagged_data.append((doc, [f"DOC_{i}"]))

# Step 3: Initialize Model Parameters
vector_size = 20  # Dimensionality of the document vectors
learning_rate = 0.01
epochs = 50

# Initialize document and word vectors randomly
doc_vectors = np.random.rand(len(tagged_data), vector_size)
word_vectors = np.random.rand(vocab_size, vector_size)

# Step 4: Train Doc2Vec Model
for epoch in range(epochs):
    for doc, tags in tagged_data:
        doc_vec_sum = np.zeros(vector_size)
        for word in doc:
            word_idx = word_to_idx[word]
            doc_vec_sum += word_vectors[word_idx]

        doc_idx = int(tags[0].split('_')[1])
        doc_vectors[doc_idx] += learning_rate * doc_vec_sum / len(doc)

# Step 5: Normalize Document Vectors
doc_vectors_normalized = normalize(doc_vectors, norm='l2')

# Step 6: Print Document Vectors
for i, doc in enumerate(data):
    print("Document", i+1, ":", doc)
    print("Vector:", doc_vectors_normalized[i])
    print()