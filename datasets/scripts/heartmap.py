import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Создание heatmap для проверки кластеров...")

# 1. Загружаем данные
df = pd.read_csv('final_clusters_with_descriptions.csv')
S = np.load('S_matrix.npy')  # матрица сходства

# 2. Создаем папку для heatmap
heatmap_dir = "heatmaps_for_clusters"
if not os.path.exists(heatmap_dir):
    os.makedirs(heatmap_dir)

# 3. Анализируем каждый кластер
clusters = df['cluster'].unique()
print(f"Всего кластеров: {len(clusters)}")

for cluster_id in clusters:
    # Получаем термины этого кластера
    cluster_terms = df[df['cluster'] == cluster_id]
    
    # Если кластер слишком маленький, пропускаем
    if len(cluster_terms) < 2:
        continue
    
    print(f"Обрабатываю кластер {cluster_id} ({len(cluster_terms)} терминов)")
    
    # Получаем индексы терминов
    indices = cluster_terms.index.tolist()
    
    # Извлекаем подматрицу сходства для этого кластера
    cluster_S = S[indices, :][:, indices]
    
    # Создаем heatmap
    plt.figure(figsize=(10, 8))
    
    # Маска для диагонали (скрываем самосходство = 1)
    mask = np.eye(len(cluster_S), dtype=bool)
    
    # Визуализация
    im = plt.imshow(cluster_S, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, label='Семантическое сходство')
    
    # Настройки графика
    plt.title(f'Heatmap кластера {cluster_id}\n'
              f'Размер: {len(cluster_terms)} терминов\n'
              f'Среднее сходство: {cluster_S.mean():.3f}')
    
    # Убираем подписи осей (слишком много)
    plt.xticks([])
    plt.yticks([])
    
    # Сохраняем
    plt.tight_layout()
    plt.savefig(f'{heatmap_dir}/heatmap_cluster_{cluster_id}.png', dpi=200, bbox_inches='tight')
    plt.close()

print(f"\nHeatmap'ы сохранены в папке '{heatmap_dir}/'")

# 4. Создаем heatmap для топ-3 самых больших кластеров
print("\nСоздаем heatmap для топ-3 самых больших кластеров...")

# Находим самые большие кластеры
cluster_sizes = df['cluster'].value_counts()
top_clusters = cluster_sizes.head(3).index.tolist()

for cluster_id in top_clusters:
    cluster_terms = df[df['cluster'] == cluster_id]
    indices = cluster_terms.index.tolist()
    cluster_S = S[indices, :][:, indices]
    
    # Создаем heatmap с подписями для маленьких кластеров
    plt.figure(figsize=(12, 10))
    
    if len(cluster_terms) <= 10:
        # Для маленьких кластеров показываем подписи
        im = plt.imshow(cluster_S, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar(im, label='Семантическое сходство')
        
        # Добавляем подписи терминов
        terms = cluster_terms['term'].tolist()
        plt.xticks(range(len(terms)), terms, rotation=90, fontsize=8)
        plt.yticks(range(len(terms)), terms, fontsize=8)
        
        plt.title(f'Кластер {cluster_id} - {len(cluster_terms)} терминов\n'
                  f'Среднее сходство: {cluster_S.mean():.3f}')
    else:
        # Для больших кластеров только heatmap
        im = plt.imshow(cluster_S, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar(im, label='Семантическое сходство')
        plt.title(f'Кластер {cluster_id} - {len(cluster_terms)} терминов')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{heatmap_dir}/heatmap_top_cluster_{cluster_id}.png', dpi=200, bbox_inches='tight')
    plt.close()

print("Готово! Проверьте папку с heatmap'ами.")