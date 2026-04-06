
# TRABAJO FINAL - PIAD-426 By Jhonatan Najarro

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. CREACIÓN DEL DATASET Y EXPORTACIÓN A CSV
# ============================================================

data = {
    "Producto": [
        "Pendrive 32GB", "Pendrive 64GB", "Pendrive 128GB",
        "Memoria RAM 8GB", "Memoria RAM 16GB", "Memoria RAM 32GB",
        "SSD 256GB", "SSD 512GB", "SSD 1TB", "Tarjeta SD 64GB",
        "Pendrive 32GB", "Memoria RAM 8GB", "SSD 256GB", "Pendrive 64GB", "SSD 1TB",
        "Memoria RAM 16GB", "Tarjeta SD 64GB", "Pendrive 128GB", "SSD 512GB", "Memoria RAM 32GB",
        "Pendrive 32GB", "SSD 256GB", "Memoria RAM 8GB", "Pendrive 64GB", "SSD 1TB",
        "Tarjeta SD 64GB", "Memoria RAM 16GB", "Pendrive 128GB", "SSD 512GB", "Memoria RAM 32GB"
    ],
    "Precio": [
        25, 45, 80, 120, 220, 380, 150, 280, 450, 35,
        25, 120, 150, 45, 450, 220, 35, 80, 280, 380,
        25, 150, 120, 45, 450, 35, 220, 80, 280, 380
    ],
    "Cantidad": [
        15, 10, 6, 5, 3, 2, 8, 4, 2, 12,
        18, 7, 11, 13, 1, 4, 20, 9, 3, 2,
        16, 6, 8, 14, 2, 11, 5, 7, 4, 3
    ],
    "Ciudad": [
        "Lima", "Cusco", "Arequipa", "Trujillo", "Lima", "Piura",
        "Lima", "Cusco", "Arequipa", "Lima", "Trujillo", "Chiclayo",
        "Huancayo", "Lima", "Ica", "Huaraz", "Cajamarca", "Puno",
        "Tarapoto", "Tacna", "Moquegua", "Tumbes", "Pucallpa", "Iquitos",
        "Chachapoyas", "Huanuco", "Cerro de Pasco", "Abancay", "Ayacucho", "Huancavelica"
    ],
    "Region": [
        "Lima", "Cusco", "Arequipa", "La Libertad", "Lima", "Piura",
        "Lima", "Cusco", "Arequipa", "Lima", "La Libertad", "Lambayeque",
        "Junin", "Lima", "Ica", "Ancash", "Cajamarca", "Puno",
        "San Martin", "Tacna", "Moquegua", "Tumbes", "Ucayali", "Loreto",
        "Amazonas", "Huanuco", "Pasco", "Apurimac", "Ayacucho", "Huancavelica"
    ]
}

# Crear DataFrame original
df_original = pd.DataFrame(data)
df_original["Ventas"] = df_original["Precio"] * df_original["Cantidad"]

# Exportar a CSV
nombre_csv = "ventas_memorikigs_nuevo.csv"
df_original.to_csv(nombre_csv, index=False)
print(f"Dataset exportado correctamente a: {nombre_csv}")


# ============================================================
# 2. IMPORTACIÓN DEL DATASET DESDE CSV
# ============================================================

df = pd.read_csv(nombre_csv)

print("=" * 55)
print(" TRABAJO FINAL PIAD-426 - MEMORYKINGS")
print("=" * 55)
print("\n=== 1. DATASET IMPORTADO DESDE CSV ===")
print(df.to_string())
print(f"\nForma del dataset  : {df.shape[0]} filas x {df.shape[1]} columnas")
print("\nTipos de datos:")
print(df.dtypes)
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nEstadísticas descriptivas:")
print(df.describe().round(2))


# ============================================================
# 3. PREPROCESAMIENTO
# ============================================================
print("\n=== 2. PREPROCESAMIENTO ===")

umbral = df["Ventas"].median()
df["AltaVenta"] = (df["Ventas"] > umbral).astype(int)
print(f"Umbral (mediana de Ventas) : S/. {umbral}")
print("Distribución AltaVenta:\n", df["AltaVenta"].value_counts().to_string())

df_encoded = pd.get_dummies(df, columns=["Ciudad", "Region", "Producto"], drop_first=True)
feat_cols = ["Precio", "Cantidad"] + [
    c for c in df_encoded.columns
    if c.startswith(("Ciudad_", "Region_", "Producto_"))
]
X = df_encoded[feat_cols]
y = df["AltaVenta"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nNormalización aplicada con StandardScaler.")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Entrenamiento: {X_train.shape[0]} muestras | Prueba: {X_test.shape[0]} muestras")


# ============================================================
# 4. MODELO DE MACHINE LEARNING (CLASIFICACIÓN)
# ============================================================
print("\n=== 3. MODELO DE CLASIFICACIÓN (Regresión Logística) ===")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Exactitud (accuracy): {acc:.3f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Baja Venta", "Alta Venta"]))

nuevo = {col: [0] for col in feat_cols}
nuevo["Precio"] = [150]
nuevo["Cantidad"] = [6]
nuevo_df = pd.DataFrame(nuevo)
nuevo_scaled = scaler.transform(nuevo_df)
pred = model.predict(nuevo_scaled)[0]
proba = model.predict_proba(nuevo_scaled)[0]

print("Predicción producto hipotético (Precio=150, Cantidad=6):")
print(f"  → Prob. Baja Venta : {proba[0]:.3f}")
print(f"  → Prob. Alta Venta : {proba[1]:.3f}")
print(f"  → Resultado        : {'ALTA VENTA' if pred == 1 else 'BAJA VENTA'}")


# ============================================================
# 5. ÁLGEBRA LINEAL CON NUMPY
# ============================================================
print("\n=== 4. ÁLGEBRA LINEAL CON NUMPY ===")

v = np.array([25, 45, 80, 120, 220])
w = np.array([15, 10, 6, 5, 3])

print("Vector Precios    (v):", v)
print("Vector Cantidades (w):", w)

prod_punto = np.dot(v, w)
print(f"\nProducto punto v·w = {prod_punto}")

norma_v = np.linalg.norm(v)
norma_w = np.linalg.norm(w)
print(f"Norma del vector Precios    : {norma_v:.4f}")
print(f"Norma del vector Cantidades : {norma_w:.4f}")

A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])
sol = np.linalg.solve(A, b)

print("\nSistema de ecuaciones:")
print("  2x + y  = 5")
print("   x + 3y = 7")
print(f"Solución: x = {sol[0]:.4f}, y = {sol[1]:.4f}")
print("Verificación Ax = b:", "✓ correcto" if np.allclose(A @ sol, b) else "✗ error")


# ============================================================
# 6. ANÁLISIS ESTADÍSTICO
# ============================================================
print("\n=== 5. ANÁLISIS ESTADÍSTICO (Ventas S/.) ===")

ventas = df["Ventas"].values
media = np.mean(ventas)
mediana = np.median(ventas)

print(f"Media   : S/. {media:.2f}")
print(f"Mediana : S/. {mediana:.2f}")


# ============================================================
# 7. VARIANZA Y DESVIACIÓN ESTÁNDAR
# ============================================================
print("\n=== 6. VARIANZA Y DESVIACIÓN ESTÁNDAR ===")

def varianza_manual(x):
    m = np.mean(x)
    return np.mean((x - m) ** 2)

def desviacion_manual(x):
    return np.sqrt(varianza_manual(x))

var_m = varianza_manual(ventas)
std_m = desviacion_manual(ventas)
var_np = np.var(ventas)
std_np = np.std(ventas)

print(f"Varianza   (manual) : {var_m:.4f}")
print(f"Desv. std  (manual) : {std_m:.4f}")
print(f"Varianza   (NumPy)  : {var_np:.4f}")
print(f"Desv. std  (NumPy)  : {std_np:.4f}")
print("¿Coinciden?", " Sí" if np.isclose(var_m, var_np) else " No")


# ============================================================
# 8. VALORES ATÍPICOS
# ============================================================
print("\n=== 7. VALORES ATÍPICOS (media ± 2·std) ===")

lim_sup = media + 2 * std_np
lim_inf = media - 2 * std_np
outliers = df[(df["Ventas"] > lim_sup) | (df["Ventas"] < lim_inf)]

print(f"Límite superior : S/. {lim_sup:.2f}")
print(f"Límite inferior : S/. {lim_inf:.2f}")
print(f"Productos atípicos encontrados: {len(outliers)}")
if len(outliers) > 0:
    print(outliers[["Producto", "Ciudad", "Precio", "Cantidad", "Ventas"]].to_string(index=False))


# ============================================================
# 9. GRÁFICOS CON MATPLOTLIB Y SEABORN
# ============================================================
print("\n=== 8. GENERANDO GRÁFICOS ===")
sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(ventas, bins=10, kde=True, color="#27AE60", edgecolor="black")
plt.axvline(media, color="red", linestyle="--", linewidth=1.8, label=f"Media: S/.{media:.0f}")
plt.axvline(mediana, color="orange", linestyle="--", linewidth=1.8, label=f"Mediana: S/.{mediana:.0f}")
plt.title("Histograma de Ventas - MemoryKings", fontsize=13)
plt.xlabel("Ventas (S/.)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.savefig("hist_ventas.png", dpi=120)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Precio",
    y="Ventas",
    hue="AltaVenta",
    palette={0: "#E74C3C", 1: "#27AE60"},
    s=100,
    alpha=0.85
)
plt.title("Precio vs Ventas  (0=Baja Venta | 1=Alta Venta)", fontsize=13)
plt.xlabel("Precio (S/.)")
plt.ylabel("Ventas (S/.)")
plt.legend(title="AltaVenta")
plt.tight_layout()
plt.savefig("scatter_precio_ventas.png", dpi=120)
plt.show()

ventas_ciudad = df.groupby("Ciudad")["Ventas"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=ventas_ciudad.index, y=ventas_ciudad.values, palette="viridis")
plt.title("Ventas Totales por Ciudad - MemorikIGs", fontsize=13)
plt.xlabel("Ciudad")
plt.ylabel("Ventas Totales (S/.)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("barras_ventas_ciudad.png", dpi=120)
plt.show()

print("Gráficos guardados: hist_ventas.png | scatter_precio_ventas.png | barras_ventas_ciudad.png")


# ============================================================
# 10. ANÁLISIS DE RESULTADOS
# ============================================================
print("\n=== 9. ANÁLISIS DE RESULTADOS ===")
print(f"1. Producto con mayores ventas : {df.loc[df['Ventas'].idxmax(), 'Producto']} — S/. {df['Ventas'].max()}")
print(f"2. Ciudad con mayor volumen    : {ventas_ciudad.idxmax()} — S/. {ventas_ciudad.max()}")
print(f"3. Media de ventas             : S/. {media:.2f}")
print(f"4. Desviación estándar         : S/. {std_np:.2f} → variación significativa en ventas")
print(f"5. Exactitud del modelo ML      : {acc:.3f} ({int(acc * 100)}% aciertos)")
print(f"6. Valor atípico detectado     : {outliers['Producto'].values[0] if len(outliers) > 0 else 'Ninguno'} con S/. {outliers['Ventas'].values[0] if len(outliers) > 0 else '-'}")

print("\n ==================== finalizado ====================")