import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 1. Cargar el dataset
# =========================
df = pd.read_csv("cars.csv")

# =========================
# 2. Manejo de valores NaN
# =========================
# Eliminar filas donde 'selling_price' sea NaN, ya que es la variable objetivo
df.dropna(subset=['selling_price'], inplace=True)

# Rellenar valores numéricos NaN con la media
numeric_cols = ['year', 'km_driven']
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))

# Rellenar valores categóricos con la moda
categorical_cols = ['owner', 'transmission', 'seller_type', 'fuel']
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# =========================
# 3. Preprocesamiento de características
# =========================
def convert_owner(owner):
    owner_str = str(owner).lower()
    if "first" in owner_str:
        return 1
    elif "second" in owner_str:
        return 2
    elif "third" in owner_str:
        return 3
    else:
        return 4  # Para "Fourth & Above" u otros

df['owner_num'] = df['owner'].apply(convert_owner)
df['transmission_num'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})
df['seller_type_num'] = df['seller_type'].map({'Individual': 0, 'Dealer': 1})
fuel_mapping = {cat: i for i, cat in enumerate(df['fuel'].unique())}
df['fuel_num'] = df['fuel'].map(fuel_mapping)

features = ['year', 'selling_price', 'km_driven', 'owner_num', 'transmission_num', 'seller_type_num', 'fuel_num']

# =========================
# 4. Determinar la línea de corte en precio con KMeans
# =========================
prices = df[['selling_price']].values
kmeans_price = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_price.fit(prices)
df['price_cluster'] = kmeans_price.labels_

# Calcular el umbral de precio
cluster_avg = df.groupby('price_cluster')['selling_price'].mean()
high_price_cluster = cluster_avg.idxmax()
centers = kmeans_price.cluster_centers_
threshold = np.mean(centers)
df_high = df[df['price_cluster'] == high_price_cluster]

# =========================
# 5. Importancia de variables en el precio
# =========================
X = df[['year', 'km_driven', 'owner_num', 'transmission_num', 'seller_type_num', 'fuel_num']]
y = df['selling_price']

# Confirmar que no hay valores NaN
if X.isnull().sum().sum() > 0:
    print("Error: X contiene valores NaN después del preprocesamiento")
    print(X.isnull().sum())
    exit()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

# Convertir a DataFrame
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nImportancia de características en el precio de venta:")
print(feature_importance_df)

# =========================
# 6. PCA para visualizar todas las variables
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Gráfico PCA
plt.figure(figsize=(10,6))
for cluster in df['price_cluster'].unique():
    subset = df[df['price_cluster'] == cluster]
    plt.scatter(subset['PCA1'], subset['PCA2'], alpha=0.6, label=f'Cluster {cluster}')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Proyección PCA de Autos Agrupados por Precio')
plt.legend()
plt.show()

# =========================
# 7. Gráfico 3D con las 3 características más importantes
# =========================
top_features = feature_importance_df['Feature'][:3].values
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
for cluster in df['price_cluster'].unique():
    subset = df[df['price_cluster'] == cluster]
    ax.scatter(subset[top_features[0]], subset[top_features[1]], subset[top_features[2]], alpha=0.6, label=f'Cluster {cluster}')
ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_zlabel(top_features[2])
plt.title('Autos Agrupados por Precio (3D)')
plt.legend()
plt.show()

# =========================
# 8. Mapa de calor de correlaciones
# =========================
plt.figure(figsize=(8,6))
sns.heatmap(df[['selling_price', 'year', 'km_driven', 'owner_num', 'transmission_num', 'seller_type_num', 'fuel_num']].corr(), 
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlaciones entre Variables")
plt.show()
