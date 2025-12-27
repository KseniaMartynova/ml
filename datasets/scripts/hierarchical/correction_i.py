import pandas as pd
df = pd.read_csv('final_clusters_with_descriptions.csv')

if len(df.columns) == 3:
    # Переименовываем колонки
    df.columns = ['term', 'cluster', 'description']
    df = df[['term', 'description', 'cluster']]
    print("переименованo")
else:
    print("Yе 3 колонки в файле!!!!!!")

output_file = 'semantic_clusters_hierarchical.xlsx'
df.to_excel(output_file, index=False)

df.to_csv('semantic_clusters_hierarchical.csv', index=False)
