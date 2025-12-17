import pandas as pd
import numpy as np
import torch
from semantic_model import get_semantic_matrix

df = pd.read_csv('combined_terms.csv')
all_terms = sorted(set(df["term"]))

with open('all_terms.csv', 'w') as f:
    for term in all_terms:
        f.write(term + '\n')

model = torch.load('hig2vec_human_200dim.pth', map_location='cpu')

embeddings = model['embeddings']
if isinstance(embeddings, torch.Tensor):
    embeddings = embeddings.numpy()

objects = model['objects']
embedding_dict = {objects[i]: embeddings[i] for i in range(len(objects))}


X = get_semantic_matrix(all_terms, embedding_dict)
print(f"Матрица X создана: {X.shape}")

np.save('X_matrix.npy', X)




