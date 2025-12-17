import numpy as np
from typing import List, Dict

def get_semantic_matrix(term_list: List[str],
                        embedding_dict: Dict[str, np.ndarray],
                        fill_value: float = 0.0) -> np.ndarray:
    """
    Возвращает матрицу эмбеддингов размера |T| x d,
    где каждая строка соответствует GO-термину из term_list.

    Parameters
    ----------
    term_list : List[str]
        Список GO-терминов (например "GO:0005737").

    embedding_dict : Dict[str, np.ndarray]
        Словарь: ключ — GO-ID, значение — вектор эмбеддинга Hig2Vec.

    fill_value : float
        Значение, которым заполняются строки для терминов,
        отсутствующих в embedding_dict.

    Returns
    -------
    X : np.ndarray
        Матрица эмбеддингов размера |T| x d.
    """

    # Определяем размерность пространства Hig2Vec
    # ищем первый доступный эмбеддинг
    example_vec = None
    for t in term_list:
        if t in embedding_dict:
            example_vec = embedding_dict[t]
            break

    if example_vec is None:
        # Ни один термин не представлен в embedding_dict
        return np.zeros((len(term_list), 0))

    d = example_vec.shape[0]
    n = len(term_list)

    # Пустая матрица эмбеддингов
    X = np.full((n, d), fill_value, dtype=float)

    # Заполняем строки
    for i, term in enumerate(term_list):
        if term in embedding_dict:
            X[i] = embedding_dict[term]

    return X