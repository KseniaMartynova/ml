import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

S = np.load('S_matrix.npy')

# Преобразуем в матрицу расстояний
D = 1 - S  # расстояние = 1 - сходство
print(f"   Матрица расстояний D: от {D.min():.4f} до {D.max():.4f}")
#  Преобразуем в сжатый формат (condensed)
D_condensed = squareform(D, checks=False)
print(f"   Вектор расстояний: {D_condensed.shape} элементов")

# иерархическую кластеризацию average
Z = linkage(D_condensed, method='average')
print(f"   Матрица связи Z: {Z.shape}")
np.save('linkage_matrix_Z.npy', Z)
np.save('distance_matrix_D.npy', D)
plt.figure(figsize=(15, 8))

from scipy.cluster.hierarchy import dendrogram
dendrogram(
    Z,
    truncate_mode='lastp',  
    p=30,  
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



