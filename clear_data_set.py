import pandas as pd

file_path = 'dataset_clients.csv'
df = pd.read_csv(file_path)

df.drop('Indice Satisfaction', axis=1, inplace=True)

nouveau_fichier_path = 'dataset_clients_sans_indice.csv'
df.to_csv(nouveau_fichier_path, index=False)
