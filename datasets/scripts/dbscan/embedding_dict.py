import pandas as pd
import torch
import numpy as np
from semantic_model import get_semantic_matrix
df = pd.read_csv('combined_terms.csv')
all_terms = sorted(set(df["term"]))
descriptions = df.groupby('term')['description'].first().loc[all_terms].values
try:
    import torch
    model_data = torch.load('hig2vec_human_200dim.pth', map_location='cpu')
    
    # Извлекаем эмбеддинги
    objects = model_data['objects']
    embeddings = model_data['embeddings']
    
    # Преобразуем в numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings
    
    # Создаем словарь embedding_dict
    embedding_dict = {objects[i]: embeddings_np[i] for i in range(len(objects))}
    print("Словарь создан")
except Exception as e:
    print("Словарь не создан")
    exit()
