# Trabajo Final PIAD-426

Análisis y desarrollo de algoritmos para Inteligencia Artificial con fundamento matemático, usando un caso práctico de ventas de productos de almacenamiento de la empresa ficticia **MemoriKings**.

## Descripción general

El proyecto integra:

- **Creación e importación de un dataset** de ventas (`ventas_memorikigs_nuevo.csv`) con columnas: Producto, Precio, Cantidad, Ciudad, Región y Ventas (Precio × Cantidad).
- **Análisis exploratorio** con Pandas: tipos de datos, valores nulos y estadísticas descriptivas.
- **Preprocesamiento para Machine Learning**: creación de la variable objetivo `AltaVenta`, codificación one-hot de variables categóricas y normalización de características numéricas.

## Machine Learning

- Formulación de un problema de **clasificación binaria**: `AltaVenta` (1 = alta venta, 0 = baja venta) a partir de la mediana de las ventas.
- División del dataset en entrenamiento y prueba (`train_test_split`).
- Entrenamiento de un modelo de **Regresión Logística** (`scikit-learn`).
- Evaluación con **accuracy** y reporte de clasificación.
- Predicción para un producto nuevo (Precio y Cantidad).

## Fundamento matemático

Usando **NumPy**:

- Definición de vectores de precios y cantidades.
- Cálculo de **producto punto** y **norma** de vectores.
- Resolución de un **sistema de ecuaciones lineales** \(2x + y = 5\), \(x + 3y = 7\).
- Cálculo de **media**, **mediana**, **varianza** y **desviación estándar** de las ventas:
  - Implementación manual.
  - Validación con funciones de NumPy.

## Estadística, outliers y visualización

- Detección de **valores atípicos** (outliers) usando el criterio:`media ± 2 * desviación estándar` sobre la columna Ventas.
- Visualizaciones con **Matplotlib** y **Seaborn**:
  - Histograma de la distribución de Ventas (con media y mediana).
  - Dispersión Precio vs Ventas coloreando por `AltaVenta`.
  - Barras de Ventas totales por Ciudad.

## Tecnologías utilizadas

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Autor

Este trabajo final fue desarrollado por:

**Jhonatan Jhon Najarro Mejia**
Estudiante de **Fundamentos y Algoritmia para Inteligencia Artificial (PIAD-426)**
SENATI
Año académico: 2026
