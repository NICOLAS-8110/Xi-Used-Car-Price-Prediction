<img width="1280" height="640" alt="WhatsApp Image 2026-11-17 at 3 00 46 PM (13)" src="https://github.com/user-attachments/assets/318b9334-031a-4640-832b-791745a8f3f5" />

# 🚗 Ξ Proyecto Xi | Predicción de Precios de Autos Usados (Rusty Bargain)

## 📋 Descripción
El servicio de venta de autos usados **Rusty Bargain** está desarrollando una aplicación para atraer nuevos clientes permitiéndoles averiguar rápidamente el valor de mercado de su coche a partir de sus especificaciones técnicas y equipamiento. 

El objetivo principal de este proyecto consiste en construir y comparar modelos de **Machine Learning** evaluando tres criterios fundamentales para el negocio:
1. **Calidad de la predicción** (medida mediante **RECM / RMSE**).
2. **Tiempo requerido para el entrenamiento**.
3. **Velocidad de inferencia (predicción)**.

## 🎯 Objetivos
- ✅ Explorar y depurar el historial de 354,369 vehículos usados, filtrando rangos anómalos de año de registro (`1900–2016`) y potencia (`45–500 CV`), e imputando valores ausentes en variables categóricas (`unknown`).
- ✅ Preparar los datos con técnicas de codificación adecuadas según cada algoritmo (**One-Hot Encoding**, **OrdinalEncoder** y tipo nativo `category`).
- ✅ Entrenar y comparar una prueba de cordura (**Regresión Lineal**), **Bosque Aleatorio (Random Forest)** y **LightGBM**.
- ✅ Evaluar el equilibrio entre precisión predictiva (**RECM**) y tiempos de cómputo (entrenamiento y predicción).
- ✅ Seleccionar el modelo óptimo para implementar en la aplicación de **Rusty Bargain**.

## 🛠️ Herramientas y Habilidades Utilizadas
* **Lenguaje:** Python
* **Librerías:** Pandas, NumPy, Scikit-Learn y LightGBM
* **Entorno:** Jupyter Notebook (`Proyecto_Xi_Prediccion_Precios_Autos_Usados.ipynb`)
* **Técnicas:** Limpieza y filtrado de anomalías, One-Hot Encoding (OHE), Ordinal Encoding, Manejo nativo de variables categóricas (`category`), GridSearchCV, Gradient Boosting y Perfilado de tiempos (`%%time`).

## 📊 Fuente de Datos y Variables
El conjunto de datos original (`car_data.csv` / `car_data.7z`) cuenta con **354,369 registros** y 16 columnas. Tras depurar registros corruptos o irreales, se trabajó con un dataset limpio de **300,243 vehículos** y las siguientes características relevantes:
* **Características técnicas y de uso:** `VehicleType` (tipo de vehículo), `Brand` (marca), `Model` (modelo), `Gearbox` (caja de cambios), `Power` (potencia en CV), `FuelType` (combustible), `Mileage` (kilometraje) y `NotRepaired` (estado de reparación).
* **Variables temporales:** `RegistrationYear` (año de registro) y `RegistrationMonth` (mes de registro).
* **Variable objetivo:** `Price` (valor de mercado en euros).

## ⚙️ Metodología
1. **Preparación y Limpieza de Datos:** Filtrado de valores imposibles en `RegistrationYear` y `Power` (conservando ~85% de los registros válidos), imputación de valores nulos categóricos como `'unknown'` y eliminación de columnas irrelevantes (`DateCrawled`, `DateCreated`, `LastSeen`, `NumberOfPictures`, `PostalCode`).
2. **División de Datos:** Partición **75% Entrenamiento / 25% Validación** (`random_state=42`).
3. **Preprocesamiento Específico por Modelo:**
   * **Regresión Lineal:** Codificación *One-Hot* (`pd.get_dummies`).
   * **Bosque Aleatorio:** Codificación ordinal (`OrdinalEncoder`) y optimización de hiperparámetros mediante `GridSearchCV` (`cv=3`).
   * **LightGBM:** Conversión directa de columnas categóricas al tipo nativo `category` y evaluación de múltiples combinaciones de hiperparámetros.

## 💡 Comparación de Resultados y Tiempos de Ejecución
| Modelo Evaluado | Codificación | RECM (RMSE) | Tiempo Entrenamiento | Tiempo Predicción |
| :--- | :---: | :---: | :---: | :---: |
| **Regresión Lineal (Prueba de cordura)** | One-Hot (OHE) | 2,606.30 € | 8.47 s | **220 ms** |
| **Bosque Aleatorio (`max_depth=15, n_estimators=100`)** | OrdinalEncoder | 1,695.17 € | 18.60 s *(2m 37s GridSearch)* | 509 ms |
| **LightGBM (`n_estimators=100, lr=0.05, num_leaves=63`)** | Nativa (`category`) | **1,683.67 €** | **3.11 s** | 369 ms |

### ⚙️ Configuración del Modelo Ganador (`LightGBM`)
```python
LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42
)
