# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Chargement des données
data = pd.read_csv('dataset_marketing_grand.csv')

# Sélection des variables pour le modèle
X = data[['Budget Réseaux Sociaux', 'Budget Télévision', 'Budget Radio']]  # Variables indépendantes
y = data['Ventes']  # Variable dépendante

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construction et entraînement du modèle de régression linéaire multiple
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Affichage des métriques d'évaluation
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Visualisation des prédictions vs valeurs réelles
plt.scatter(y_test, y_pred)
plt.xlabel('Ventes Réelles')
plt.ylabel('Ventes Prédites')
plt.title('Comparaison des Ventes Réelles et Prédites')
plt.show()