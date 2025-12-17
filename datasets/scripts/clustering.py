import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

S = np.load('S_matrix.npy')

# 2. Преобразуем в матрицу расстояний
print("\n2. Преобразуем в матрицу расстояний...")
D = 1 - S  # расстояние = 1 - сходство
print(f"   Матрица расстояний D: от {D.min():.4f} до {D.max():.4f}")

# 3. Преобразуем в сжатый формат (condensed)
print("\n3. Преобразуем в сжатый формат для linkage...")
D_condensed = squareform(D, checks=False)
print(f"   Вектор расстояний: {D_condensed.shape} элементов")

# 4. Выполняем иерархическую кластеризацию
print("\n4. Выполняем иерархическую кластеризацию...")
print("   Используем метод: 'average'")
Z = linkage(D_condensed, method='average')
print(f"   Матрица связи Z: {Z.shape}")

# 5. Сохраняем результаты
print("\n5. Сохраняем результаты...")
np.save('linkage_matrix_Z.npy', Z)
np.save('distance_matrix_D.npy', D)

print("   Сохранено:")
print("   - linkage_matrix_Z.npy (матрица связи)")
print("   - distance_matrix_D.npy (матрица расстояний)")

# 6. Создаем упрощенную дендрограмму
print("\n6. Создаем дендрограмму...")
plt.figure(figsize=(15, 8))

from scipy.cluster.hierarchy import dendrogram
dendrogram(
    Z,
    truncate_mode='lastp',  # показываем только последние p кластеров
    p=30,  # показываем 30 последних слияний
    show_leaf_counts=True,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.title('Дендрограмма GO-терминов (метод: average)', fontsize=16, fontweight='bold')
plt.xlabel('GO-термины', fontsize=12)
plt.ylabel('Расстояние', fontsize=12)
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
plt.savefig('dendrogram.pdf', bbox_inches='tight')
print("   Дендрограмма сохранена в dendrogram.png и dendrogram.pdf")

# 7. Проверяем структуру матрицы Z
print("\n7. Структура матрицы связи Z:")
print(f"   Форма: {Z.shape}")
print("   Первые 5 строк матрицы Z:")
for i in range(min(5, len(Z))):
    print(f"   {i}: кластеры {int(Z[i,0])} и {int(Z[i,1])} объединяются на расстоянии {Z[i,2]:.4f}, размер нового кластера: {int(Z[i,3])}")

