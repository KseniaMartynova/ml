import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
X = np.load('X_matrix.npy')
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

np.save('X_2d_simple.npy', X_2d)

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=20)
plt.title('t-SNE проекция GO-терминов')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.savefig('tSNE_simple.png', dpi=150, bbox_inches='tight')
plt.show()
