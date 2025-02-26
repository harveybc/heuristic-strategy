import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
#                          1. CARGA Y LIMPIEZA DE DATOS                       #
###############################################################################

# Lee el dataset desde un archivo CSV
df = pd.read_csv("Car_Datasets_inventory.csv")

# Elimina duplicados si los hubiera
df.drop_duplicates(inplace=True)

# Verificar que las columnas críticas existan
required_cols = ["selling_price", "km_driven", "year", "owner", "transmission", "seller_type", "fuel"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"La columna {col} es obligatoria y no está en el CSV.")

# Elimina filas con NaN en la variable objetivo 'selling_price'
df.dropna(subset=["selling_price"], inplace=True)

# Para las columnas numéricas 'year' y 'km_driven', se imputan los valores faltantes con la media
numeric_cols = ["year", "km_driven"]
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# Para las columnas categóricas, se imputa la moda
categorical_cols = ["owner", "transmission", "seller_type", "fuel"]
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

###############################################################################
#                      2. PREPROCESAMIENTO DE CARACTERÍSTICAS                #
###############################################################################

def convert_owner(owner_str):
    """Convierte la columna 'owner' en un valor ordinal: First->1, Second->2, Third->3, otros->4."""
    try:
        o = str(owner_str).lower()
    except:
        return 4
    if "first" in o:
        return 1
    elif "second" in o:
        return 2
    elif "third" in o:
        return 3
    else:
        return 4

df["owner_num"] = df["owner"].apply(convert_owner)

# Para 'transmission': Manual=0, Automatic=1
df["transmission_num"] = df["transmission"].map({"Manual": 0, "Automatic": 1})
df["transmission_num"].fillna(df["transmission_num"].mode()[0], inplace=True)

# Para 'seller_type': Individual=0, Dealer=1
df["seller_type_num"] = df["seller_type"].map({"Individual": 0, "Dealer": 1})
df["seller_type_num"].fillna(df["seller_type_num"].mode()[0], inplace=True)

# Para 'fuel': asignar un entero arbitrario (no existe orden natural)
fuel_mapping = {cat: i for i, cat in enumerate(df["fuel"].unique())}
df["fuel_num"] = df["fuel"].map(fuel_mapping)
df["fuel_num"].fillna(df["fuel_num"].mode()[0], inplace=True)

# Verificar que en las columnas de interés no queden NaN; si es así, imputar con la mediana
cols_to_check = ["year", "km_driven", "owner_num", "transmission_num", "seller_type_num", "fuel_num"]
for col in cols_to_check:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Lista de características de interés (todas)
features = ["year", "selling_price", "km_driven", "owner_num", "transmission_num", "seller_type_num", "fuel_num"]

###############################################################################
#                  3. SEGMENTACIÓN DE PRECIO CON K-MEANS (k=2)                #
###############################################################################

# Aplicamos KMeans con k=2 sobre 'selling_price'
prices = df[["selling_price"]].values
kmeans_price = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_price.fit(prices)
df["price_cluster"] = kmeans_price.labels_

# Calcular el promedio de 'selling_price' en cada cluster y determinar el cluster de precio alto
cluster_avg = df.groupby("price_cluster")["selling_price"].mean()
high_price_cluster = cluster_avg.idxmax()  # Este será el "cluster de caros"
# El otro se considerará "cluster de baratos"

# Definir el umbral de precio como el promedio de los dos centros
centers = kmeans_price.cluster_centers_
threshold = np.mean(centers)
print(f"Umbral de precio determinado automáticamente: {threshold:.2f}")

# Filtrar el subconjunto de autos de precio alto
df_high = df[df["price_cluster"] == high_price_cluster]

###############################################################################
#                4. IMPORTANCIA DE VARIABLES EN EL PRECIO (R.F.)              #
###############################################################################

X = df[["year", "km_driven", "owner_num", "transmission_num", "seller_type_num", "fuel_num"]]
y = df["selling_price"]

if X.isnull().sum().sum() > 0:
    raise ValueError("Existen valores NaN en las variables independientes después del preprocesamiento.")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nImportancia de características en el precio de venta:")
print(feature_importance_df)

###############################################################################
# 5. REDUCCIÓN DIMENSIONAL CON PCA (2D para visualización)
###############################################################################

cols_for_pca = ["year", "km_driven", "owner_num", "transmission_num", "seller_type_num", "fuel_num"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[cols_for_pca])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

###############################################################################
# 6. GRÁFICOS
###############################################################################

# Definir etiquetas personalizadas para los clusters
def cluster_label(cluster_id):
    if cluster_id == high_price_cluster:
        return "Cluster de Caros"
    else:
        return "Cluster de Baratos"

# 6.1 Histograma de selling_price con línea de corte
plt.figure(figsize=(10,6))
plt.hist(df["selling_price"], bins=50, color="skyblue", edgecolor="black", alpha=0.7)
plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Umbral = ${threshold:,.0f}")
plt.title("Distribución de Precio de Venta de Autos Usados")
plt.xlabel("Precio de Venta (USD)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.show()

# 6.2 Scatter plot: km_driven vs. selling_price, coloreado según el cluster de precio
plt.figure(figsize=(10,6))
for cluster_id in df["price_cluster"].unique():
    subset = df[df["price_cluster"] == cluster_id]
    plt.scatter(subset["km_driven"], subset["selling_price"], alpha=0.6, 
                label=cluster_label(cluster_id))
median_km_high = np.median(df_high["km_driven"])
plt.axvline(x=median_km_high, color="green", linestyle="--", linewidth=2,
            label="Mediana km_driven (autos caros)")
plt.xlabel("Kilómetros Recorridos")
plt.ylabel("Precio de Venta (USD)")
plt.title("Autos agrupados por Clúster de Precio (k=2)")
plt.legend()
plt.tight_layout()
plt.show()

# 6.3 Scatter plot PCA 2D: Proyección de autos por cluster
plt.figure(figsize=(10,6))
for cluster_id in df["price_cluster"].unique():
    subset = df[df["price_cluster"] == cluster_id]
    plt.scatter(subset["PCA1"], subset["PCA2"], alpha=0.6, label=cluster_label(cluster_id))
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Proyección PCA (2D) de Autos Agrupados por Precio")
plt.legend()
plt.tight_layout()
plt.show()

# 6.4 Gráfico 3D con las 3 variables más importantes (según RandomForest)
top_features = feature_importance_df["Feature"].head(3).values
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")
for cluster_id in df["price_cluster"].unique():
    subset = df[df["price_cluster"] == cluster_id]
    ax.scatter(subset[top_features[0]],
               subset[top_features[1]],
               subset[top_features[2]],
               alpha=0.6,
               label=cluster_label(cluster_id))
ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_zlabel(top_features[2])
plt.title("Visualización 3D de las 3 Variables más Influyentes (Clusters de Precio)")
plt.legend()
plt.tight_layout()
plt.show()

# 6.5 Mapa de calor de correlaciones
corr_cols = ["selling_price", "year", "km_driven", "owner_num", "transmission_num", "seller_type_num", "fuel_num"]
plt.figure(figsize=(8,6))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor: Correlaciones entre Variables")
plt.tight_layout()
plt.show()

###############################################################################
# 7. CÁLCULO DEL CENTROIDE DEL CLUSTER DE AUTOS CAROS
###############################################################################

# Para variables numéricas: calcular la media en el cluster de autos caros
num_cols_for_centroid = ["year", "selling_price", "km_driven", "owner_num", "transmission_num", "seller_type_num", "fuel_num"]
centroid_numeric = df_high[num_cols_for_centroid].mean()

# Para las variables categóricas originales: usar la moda
cat_cols_for_centroid = ["owner", "transmission", "seller_type", "fuel"]
centroid_categorical = {
    "owner": df_high["owner"].mode().iloc[0],
    "transmission": df_high["transmission"].mode().iloc[0],
    "seller_type": df_high["seller_type"].mode().iloc[0],
    "fuel": df_high["fuel"].mode().iloc[0]
}

print("\n=== Centroide del Cluster de Autos Caros ===")
print("Valores promedio (variables numéricas):")
print(centroid_numeric)
print("\nCategorías predominantes (variables originales):")
print(centroid_categorical)
