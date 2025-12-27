import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('final_clusters_with_descriptions.csv')
S = np.load('S_matrix.npy') 
heatmap_dir = "heatmaps_for_clusters"
if not os.path.exists(heatmap_dir):
    os.makedirs(heatmap_dir)


clusters = df['cluster'].unique()
print(f"Всего кластеров: {len(clusters)}")

for cluster_id in clusters:
    # Получаем термины этого кластера
    cluster_terms = df[df['cluster'] == cluster_id]
    
    # Если кластер слишком маленький, пропускаем
    if len(cluster_terms) < 2:
        continue

    
    # Получаем индексы терминов
    indices = cluster_terms.index.tolist()
    
    # Извлекаем подматрицу сходства для этого кластера
    cluster_S = S[indices, :][:, indices]
    
    # Создаем heatmap
    plt.figure(figsize=(10, 8))
    
    # Маска для диагонали (скрываем самосходство = 1)
    mask = np.eye(len(cluster_S), dtype=bool)
    im = plt.imshow(cluster_S, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, label='Семантическое сходство')
    
    plt.title(f'Heatmap кластера {cluster_id}\n'
              f'Размер: {len(cluster_terms)} терминов\n'
              f'Среднее сходство: {cluster_S.mean():.3f}')
    
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{heatmap_dir}/heatmap_cluster_{cluster_id}.png', dpi=200, bbox_inches='tight')
    plt.close()
    plt.savefig(f'{heatmap_dir}/heatmap_top_cluster_{cluster_id}.png', dpi=200, bbox_inches='tight')
    plt.close()


print("Готово! Проверьте папку с heatmap'ами.")
