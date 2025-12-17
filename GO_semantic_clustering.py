import torch
import os
from modules.semantic_model import get_semantic_matrix

# измени под свой путь к файлу hig2vec_human_200dim.pth
MODEL_PATH = os.path.join('datasets','GO_semantic_models','hig2vec_human_200dim.pth')
print(MODEL_PATH)

model = torch.load(MODEL_PATH, map_location="cpu")
objects, embeddings = model['objects'], model['embeddings']

d_human_vec = {k: v for k, v in zip(objects, embeddings)}
d_human_GO = {k: v for k, v in d_human_vec.items() if "GO:" in k} # model contains only GO terms

test_terms = ['GO:0098727', 'GO:0050684', 'GO:0043484', 'GO:0031032', 'GO:0031507']



X = get_semantic_matrix(test_terms, d_human_GO)

print(f'Векторное представление термина {test_terms[2]}:\n{X[2]}')