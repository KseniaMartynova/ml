import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X_2d = np.load('X_2d_simple.npy')
terms_df = pd.read_csv('combined_terms.csv')

dbscan = DBSCAN(eps=1.8, min_samples=4, metric='euclidean')  
labels = dbscan.fit_predict(X_2d)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = np.sum(labels == -1)

print("Кластеров {n_clusters}")
print("Шум {noise_count} точек ({noise_count/len(labels)*100:.1f}%)")

results = pd.DataFrame({
    'term': terms_df['term'],
    'description': terms_df['description'],
    'cluster': labels
})
results.to_csv('final_dbscan_clusters.csv', index=False)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', s=30)
plt.colorbar(scatter, label='Кластер')
plt.title('DBSCAN кластеризация GO-терминов')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.savefig('dbscan_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
