import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Cargar el dataset
# =========================
# Supongamos que el archivo CSV se llama "Car_Datasets_inventory.csv"
df = pd.read_csv("Car_Datasets_inventory.csv")

# Eliminar filas con valores faltantes en las columnas críticas
df.dropna(subset=["selling_price", "km_driven", "year", "owner", "transmission", "seller_type", "fuel"], inplace=True)

# =========================
# 2. Preprocesamiento
# =========================
# Convertir 'owner' a valor numérico (orden ordinal)
def convert_owner(owner):
    # Convertir a cadena y pasar a minúsculas
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

# Convertir 'transmission' a 0/1 (Manual=0, Automatic=1)
df['transmission_num'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})

# Convertir 'seller_type' a 0/1 (Individual=0, Dealer=1)
df['seller_type_num'] = df['seller_type'].map({'Individual': 0, 'Dealer': 1})

# Para 'fuel', asignar números arbitrarios (no existe orden natural)
fuel_mapping = {cat: i for i, cat in enumerate(df['fuel'].unique())}
df['fuel_num'] = df['fuel'].map(fuel_mapping)

# Variables seleccionadas para el análisis
features = ['year', 'selling_price', 'km_driven', 'owner_num', 'transmission_num', 'seller_type_num', 'fuel_num']

# =========================
# 3. Determinar la línea de corte en precio usando KMeans (k=2)
# =========================
prices = df[['selling_price']].values
kmeans_price = KMeans(n_clusters=2, random_state=42)
kmeans_price.fit(prices)
df['price_cluster'] = kmeans_price.labels_

# Calcular el promedio de selling_price en cada cluster
cluster_avg = df.groupby('price_cluster')['selling_price'].mean()
high_price_cluster = cluster_avg.idxmax()  # Cluster con precio mayor en promedio

# Calcular el umbral como el promedio de los centros de ambos clusters
centers = kmeans_price.cluster_centers_
threshold = np.mean(centers)
print("Umbral de precio determinado:", threshold)

# =========================
# 4. Filtrar autos de precio alto
# =========================
df_high = df[df['price_cluster'] == high_price_cluster]

# =========================
# 5. Calcular el centroide del cluster de autos caros
# =========================
# Para variables numéricas: calcular la media
centroid_numeric = df_high[['year', 'selling_price', 'km_driven', 'owner_num', 
                              'transmission_num', 'seller_type_num', 'fuel_num']].mean()

# Para las variables categóricas originales: calcular la moda
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
# Línea vertical en la mediana de km_driven de autos caros
plt.axvline(x=np.median(df_high['km_driven']), color='green', linestyle='--', linewidth=2, 
            label='Mediana km_driven (autos caros)')
plt.xlabel('Kilómetros Recorridos')
plt.ylabel('Precio de Venta (USD)')
plt.title('Autos agrupados por Clúster de Precio')
plt.legend()
plt.show()
