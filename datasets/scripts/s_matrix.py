import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

X = np.load('X_matrix.npy')

S = cosine_similarity(X)
print(f"   Матрица S: {S.shape}")
print(f"   Значения в S: от {S.min():.4f} до {S.max():.4f}")


np.save('S_matrix.npy', S)






