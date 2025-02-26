import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Cargar el dataset
# =========================

df = pd.read_csv("cars.csv")

# Verificamos las primeras filas
print(df.head())

# =========================
# 2. Preprocesamiento
# =========================
# Convertir 'owner' a valor numérico (asumiendo un orden)
def convert_owner(owner):
    owner = owner.lower()
    if "first" in owner:
        return 1
    elif "second" in owner:
        return 2
    elif "third" in owner:
        return 3
    else:
        return 4  # "Fourth & Above" u otros

df['owner_num'] = df['owner'].apply(convert_owner)

# Convertir 'transmission' a 0/1 (Manual=0, Automatic=1)
df['transmission_num'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})

# Convertir 'seller_type' a 0/1 (Individual=0, Dealer=1)
df['seller_type_num'] = df['seller_type'].map({'Individual': 0, 'Dealer': 1})

# Para 'fuel', al no existir un orden natural, asignamos números arbitrarios
fuel_mapping = {cat: i for i, cat in enumerate(df['fuel'].unique())}
df['fuel_num'] = df['fuel'].map(fuel_mapping)

# Variables seleccionadas para el análisis: todas
features = ['year', 'selling_price', 'km_driven', 'owner_num', 'transmission_num', 'seller_type_num', 'fuel_num']

# =========================
# 3. Determinar la línea de corte en precio
# =========================
# Usamos K-means con k=2 sobre selling_price para separar autos de precio bajo y alto.
prices = df[['selling_price']].values
kmeans_price = KMeans(n_clusters=2, random_state=42)
kmeans_price.fit(prices)
df['price_cluster'] = kmeans_price.labels_

# Calcular los centros de los clusters y determinar cuál es el de precio alto
cluster_avg = df.groupby('price_cluster')['selling_price'].mean()
high_price_cluster = cluster_avg.idxmax()

# Definir el umbral como el promedio de los dos centros (puede interpretarse como la frontera)
centers = kmeans_price.cluster_centers_
threshold = np.mean(centers)
print("Umbral de precio determinado:", threshold)

# =========================
# 4. Filtrar autos de precio alto (por ejemplo, > threshold)
# =========================
df_high = df[df['price_cluster'] == high_price_cluster]
# O, si se prefiere, filtrar directamente por precio mayor a un valor (ej. > 500,000)
# df_high = df[df['selling_price'] > 500000]

# =========================
# 5. Calcular el centroide del cluster de autos caros
# =========================
# Para variables numéricas, usamos la media.
centroid_numeric = df_high[['year', 'selling_price', 'km_driven', 'owner_num', 'transmission_num', 'seller_type_num', 'fuel_num']].mean()

# Para las variables categóricas originales, usamos la moda:
centroid_categorical = {
    'fuel': df_high['fuel'].mode().iloc[0],
    'seller_type': df_high['seller_type'].mode().iloc[0],
    'transmission': df_high['transmission'].mode().iloc[0],
    'owner': df_high['owner'].mode().iloc[0]
}

print("\nCentroide (valores promedio numéricos) del cluster de autos caros:")
print(centroid_numeric)
print("\nCentroide (categorías predominantes) del cluster de autos caros:")
print(centroid_categorical)

# =========================
# 6. Generar gráficos
# =========================
# Gráfico 1: Histograma de 'selling_price' con línea de corte
plt.figure(figsize=(10,6))
plt.hist(df['selling_price'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral = ${threshold:,.0f}')
plt.title('Distribución de Precio de Venta de Autos Usados')
plt.xlabel('Precio de Venta (USD)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# Gráfico 2: Scatter plot de km_driven vs. selling_price, coloreado según el cluster de precio
plt.figure(figsize=(10,6))
for cluster in df['price_cluster'].unique():
    subset = df[df['price_cluster'] == cluster]
    plt.scatter(subset['km_driven'], subset['selling_price'], alpha=0.6, label=f'Cluster {cluster}')
plt.axvline(x=np.median(df_high['km_driven']), color='green', linestyle='--', linewidth=2, label='Mediana km_driven (autos caros)')
plt.xlabel('Kilómetros Recorridos')
plt.ylabel('Precio de Venta (USD)')
plt.title('Autos agrupados por Clúster de Precio')
plt.legend()
plt.show()
