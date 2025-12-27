import pandas as pd

df1 = pd.read_excel('Atx_enrichment_raw.xlsx')
df2 = pd.read_excel( 'Young_enrichment_raw.xlsx')
df3 = pd.read_excel('Old_enrichment_raw.xlsx')
df4 = pd.read_excel('CS_enrichment_raw.xlsx')

result = pd.concat([df1, df2, df3, df4])[['term', 'description']].drop_duplicates()
result.to_csv('combined_terms.csv', index=False)
