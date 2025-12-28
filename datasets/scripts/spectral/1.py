import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
import warnings
warnings.filterwarnings('ignore')
X = np.load('X_matrix.npy')
combined_df = pd.read_csv('combined_terms.csv')

# Создаем граф соседства для спектральной кластеризации
n_neighbors = 15  
affinity_matrix = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)  # делаем симметричным

# Подбираем число кластеров
best_score = -1
best_n_clusters = 0
best_labels = None

# Проверяем разное число кластеров
for n_clusters in range(20, 31):  # от 20 до 30 кластеров
    print(f"  Пробуем {n_clusters} кластеров...", end=" ")
    
    try:
        # Спектральная кластеризация
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed_nearest_neighbors',
            n_neighbors=n_neighbors,
            random_state=42,
            assign_labels='kmeans',
            n_init=10
        )
        
        # Используем предвычисленный граф
        labels = spectral.fit_predict(affinity_matrix)
        
        # силуэтный коэффициент на основе косинусного расстояния
        from sklearn.metrics import silhouette_score
        # Для силуэта используем косинусное расстояние
        distances = pairwise_distances(X, metric='cosine')
        if len(np.unique(labels)) > 1:
            score = silhouette_score(distances, labels, metric='precomputed')
            print(f"силуэтный score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
        else:
            print("только 1 кластер")
    except Exception as e:
        print(f"ошибка: {e}")

# Понижаем размерность для визуализации
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                      c=best_labels, 
                      cmap='tab20',
                      s=40,
                      alpha=0.8,
                      edgecolors='k',
                      linewidth=0.3)

plt.colorbar(scatter, label='ID кластера')
plt.title(f'Спектральная кластеризация GO-терминов ({best_n_clusters} кластеров)\n', fontsize=16)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
plt.grid(True, alpha=0.2)


unique, counts = np.unique(best_labels, return_counts=True)
for cluster_id, size in zip(unique, counts):
    # Находим центр кластера
    if size > 0:
        cluster_points = X_2d[best_labels == cluster_id]
        center = cluster_points.mean(axis=0)

plt.savefig("spectral_clusters_pca_visualization.png", dpi=200, bbox_inches='tight')
plt.show()



# среднее косинусное сходство внутри кластеров
from sklearn.metrics.pairwise import cosine_similarity

print("\nСреднее косинусное сходство внутри кластеров:")
good_clusters = 0
for cluster_id in unique:
    cluster_indices = np.where(best_labels == cluster_id)[0]
    if len(cluster_indices) > 1:
        X_cluster = X[cluster_indices]
        similarity_matrix = cosine_similarity(X_cluster)
        np.fill_diagonal(similarity_matrix, 0)  # игнорируем самосходство
        n_pairs = len(cluster_indices) * (len(cluster_indices) - 1)
        if n_pairs > 0:
            avg_similarity = similarity_matrix.sum() / n_pairs
            print(f"  Кластер {cluster_id}: {avg_similarity:.3f} ({len(cluster_indices)} терминов)")
            if avg_similarity > 0.7:
                good_clusters += 1
        else:
            avg_similarity = 0
    else:
        print(f"  Кластер {cluster_id}: только 1 термин, сходство не определено")


for cluster_id, size in zip(unique, counts):
    print(f"  Кластер {cluster_id}: {size} терминов ({size/len(best_labels)*100:.1f}%)")

results_df = pd.DataFrame({
    'term': combined_df['term'].values,
    'description': combined_df['description'].values,
    'cluster': best_labels
})
results_df.to_excel("semantic_clusters_spectral.xlsx", index=False)


