import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
X = np.load('X_matrix.npy')
S = cosine_similarity(X)
np.save('S_matrix.npy', S)







