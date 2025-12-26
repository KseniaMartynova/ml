import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
import os

clusters_df = pd.read_csv('final_clusters_with_descriptions.csv')
final_table = clusters_df.copy()
final_table.columns = ['GO_term', 'description', 'cluster_id']

final_table = final_table.sort_values('cluster_id')

final_table.to_csv('FINAL_RESULTS.csv', index=False, encoding='utf-8-sig')
final_table.to_excel('FINAL_RESULTS.xlsx', index=False)


plt.figure(figsize=(10, 6))
cluster_counts = final_table['cluster_id'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Номер кластера')
plt.ylabel('Количество терминов')
plt.title('Распределение GO-терминов по кластерам')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_distribution.png', dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
sorted_counts = cluster_counts.sort_values(ascending=False)
plt.bar(range(len(sorted_counts)), sorted_counts.values)
plt.xlabel('Кластеры (отсортированы по размеру)')
plt.ylabel('Количество терминов')
plt.title('Размеры кластеров (по убыванию)')
plt.xticks([])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_sizes_sorted.png', dpi=150)
plt.close()

print("1. FINAL_RESULTS.csv - таблица с терминами и кластерами")
print("2. FINAL_RESULTS.xlsx - то же в Excel")
print("3. cluster_distribution.png - визуализация")
print("4. cluster_sizes_sorted.png - визуализация")
print("\nЗадание выполнено! ✓")
