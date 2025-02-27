import pandas as pd

# Charger le fichier Excel
data_cat = pd.read_excel("AMA08001.xlsx")
data_dog= pd.read_excel("VYE15004.xlsx")
# Afficher les premières lignes
print(data_dog.head())

print(data_cat.head())