import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import os

Z = np.load('linkage_matrix_Z.npy')
S = np.load('S_matrix.npy')
df = pd.read_csv('combined_terms.csv')
all_terms = sorted(set(df["term"]))

# Функция для оценки кластеризации
def evaluate_clustering(threshold):
    clusters = fcluster(Z, t=threshold, criterion='distance')
    n_clusters = len(np.unique(clusters))
    
    # Группируем термины по кластерам
    cluster_data = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_data:
            cluster_data[cluster_id] = []
        cluster_data[cluster_id].append(i)
    
    # Оцениваем каждый кластер
    good_clusters = 0
    total_clusters_evaluated = 0
    
    for cluster_id, indices in cluster_data.items():
        if len(indices) < 2:
            continue  # Пропускаем одиночные кластеры
        
        total_clusters_evaluated += 1
        cluster_S = S[indices, :][:, indices]
        np.fill_diagonal(cluster_S, np.nan)
        mean_sim = np.nanmean(cluster_S)
        
        if mean_sim >= 0.7:
            good_clusters += 1
    
    if total_clusters_evaluated > 0:
        good_percentage = good_clusters / total_clusters_evaluated * 100
    else:
        good_percentage = 0
    
    return n_clusters, good_percentage


thresholds_to_test = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
results = []

for threshold in thresholds_to_test:
    n_clusters, good_pct = evaluate_clustering(threshold)
    results.append((threshold, n_clusters, good_pct))
    print(f"{threshold:.2f}   | {n_clusters:^9} | {good_pct:.1f}%")

# Находим лучший порог (баланс между числом кластеров и качеством)
best_threshold = None
best_score = -1

for threshold, n_clusters, good_pct in results:
    if 20 <= n_clusters <= 30:  # Целевой диапазон
        # Оценка = качество - штраф за отклонение от 25 кластеров
        score = good_pct - abs(n_clusters - 25) * 2
        if score > best_score:
            best_score = score
            best_threshold = threshold

print(f"\nЛучший порог: {best_threshold:.2f}")

# Применяем лучший порог и анализируем детально
print(f"\nДетальный анализ при пороге {best_threshold:.2f}:")
clusters = fcluster(Z, t=best_threshold, criterion='distance')
n_clusters = len(np.unique(clusters))

# Создаем DataFrame
cluster_df = pd.DataFrame({
    'term': all_terms,
    'cluster': clusters
})

# Группируем и анализируем
cluster_stats = []
cluster_groups = cluster_df.groupby('cluster')

for cluster_id, group in cluster_groups:
    size = len(group)
    if size >= 2:
        indices = group.index.tolist()
        cluster_S = S[indices, :][:, indices]
        np.fill_diagonal(cluster_S, np.nan)
        mean_sim = np.nanmean(cluster_S)
        min_sim = np.nanmin(cluster_S)
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': size,
            'mean_similarity': mean_sim,
            'min_similarity': min_sim,
            'status': 'GOOD' if mean_sim >= 0.7 else 'POOR'
        })

# Сортируем по размеру
cluster_stats.sort(key=lambda x: x['size'], reverse=True)

# Сохраняем финальные кластеры
output_file = f"final_clusters_threshold_{best_threshold:.2f}.csv"
cluster_df.to_csv(output_file, index=False)

# Создаем файл с описанием кластеров
descriptions = df.groupby('term')['description'].first().loc[all_terms].values
cluster_df['description'] = descriptions

# Сохраняем с описаниями
cluster_df.to_csv('final_clusters_with_descriptions.csv', index=False)

print(f"\nСохранено:")
print(f"1. final_clusters_threshold_{best_threshold:.2f}.csv")
print(f"2. final_clusters_with_descriptions.csv")

# Визуализация распределения кластеров
plt.figure(figsize=(10, 6))
sizes = [stat['size'] for stat in cluster_stats]
plt.bar(range(len(sizes)), sizes)
plt.xlabel('Кластеры (отсортированы по размеру)')
plt.ylabel('Количество терминов')
plt.title(f'Распределение размеров кластеров (порог={best_threshold:.2f})')
plt.tight_layout()
plt.savefig('cluster_size_distribution.png', dpi=150)
plt.show()

print("\nСоветы по интерпретации:")
print("1. Посмотрите heatmap'ы в папке cluster_heatmaps/")
print("2. Проанализируйте большие кластеры с низким сходством")

print("3. При необходимости можно разделить большие кластеры вручную")
