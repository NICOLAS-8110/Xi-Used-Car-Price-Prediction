<img width="1280" height="640" alt="WhatsApp Image 2026-11-17 at 3 00 46 PM (13)" src="https://github.com/user-attachments/assets/318b9334-031a-4640-832b-791745a8f3f5" />

# 🚗 Ξ Proyecto Xi | Predicción de Precios de Autos Usados (Rusty Bargain)

## 📋 Descripción
**Rusty Bargain** desea desarrollar una aplicación capaz de estimar el valor de mercado de vehículos usados a partir de sus características técnicas. El objetivo principal de este proyecto consiste en construir y comparar múltiples modelos de **Machine Learning** que proporcionen predicciones altamente precisas manteniendo tiempos de entrenamiento e inferencia razonables para una futura implementación en producción.

Para lograrlo, se desarrolló un flujo completo de ciencia de datos que abarcó limpieza profunda, tratamiento de valores ausentes, ingeniería de características (codificación *One-Hot* y manejo nativo de categorías) y optimización de algoritmos basados en *Gradient Boosting* evaluados mediante la métrica **RECM (RMSE)**.

## 🎯 Objetivos
- ✅ Realizar exploración, limpieza de anomalías y tratamiento de valores ausentes en el historial de vehículos usados.
- ✅ Aplicar técnicas de codificación adecuadas según el algoritmo (**One-Hot Encoding** vs. tipo nativo `category`).
- ✅ Entrenar y comparar modelos lineales, basados en árboles y ensambles avanzados (**Random Forest, CatBoost, XGBoost y LightGBM**).
- ✅ Evaluar el equilibrio entre calidad predictiva (**RMSE**) y eficiencia computacional (tiempos de entrenamiento e inferencia).
- ✅ Seleccionar el modelo óptimo listo para producción.

## 🛠️ Herramientas y Habilidades Utilizadas
* **Lenguaje:** Python
* **Librerías:** Pandas, NumPy, Scikit-Learn, LightGBM, CatBoost, XGBoost, Matplotlib y Seaborn
* **Entorno:** Jupyter Notebook ([Proyecto_Xi_Prediccion_Precios_Autos_Usados.ipynb](https://github.com/santidata7/Xi_Used_Car_Price_Prediction/blob/main/Proyecto_Xi_Prediccion_Precios_Autos_Usados.ipynb))
* **Técnicas:** Limpieza y curaduría de datos, One-Hot Encoding (OHE), Manejo nativo de variables categóricas, Gradient Boosting, Ajuste de hiperparámetros y Evaluación de tiempos de ejecución.

## 📊 Fuente de Datos y Variables
El conjunto de datos (`car_data.7z`) contiene registros históricos de vehículos usados con las siguientes variables principales:
* **Características técnicas y de uso:** `VehicleType` (tipo de vehículo), `Brand` (marca), `Model` (modelo), `Gearbox` (caja de cambios), `Power` (potencia), `FuelType` (tipo de combustible), `Kilometer` (kilometraje) y `NotRepaired` (estado de reparación).
* **Variables temporales:** `RegistrationYear` (año de registro) y `RegistrationMonth` (mes de registro).
* **Variable objetivo:** `Price` (valor de mercado en euros).

## ⚙️ Metodología
1. **Preparación y Limpieza de Datos:** Tratamiento de valores ausentes, corrección de inconsistencias, eliminación de registros anómalos (años de registro o potencias irreales) y conversión de tipos de datos.
2. **División de Datos:** Partición estructurada en conjuntos de **Entrenamiento**, **Validación** y **Prueba** para evitar fuga de datos (*data leakage*).
3. **Ingeniería de Características:** Implementación de **One-Hot Encoding (OHE)** para modelos tradicionales y aprovechamiento del procesamiento nativo mediante el tipo de dato `category` para **LightGBM** y **CatBoost**.
4. **Entrenamiento y Benchmark:** Evaluación sistemática de modelos midiendo tanto el error cuadrático medio (**RMSE**) como el tiempo de cómputo.

## 💡 Comparación de Resultados y Tiempos de Ejecución
| Modelo Evaluado | RMSE (RECM) | Tiempo Aproximado | Observaciones |
| :--- | :---: | :---: | :--- |
| **Regresor Ficticio (Dummy)** | 4,562.24 € | — | Línea base mínima de referencia |
| **Regresión Lineal** | 2,554.63 € | — | Primer modelo predictivo base |
| **Árbol de Decisión** | 1,746.37 € | — | Primer modelo no lineal evaluado |
| **Bosque Aleatorio (Random Forest)** | 1,594.89 € | ~6 min | Buen desempeño, alto costo temporal |
| **CatBoost** | 1,523.59 € | ~13 min 18 s | Alta precisión con categorías nativas |
| **XGBoost** | 1,502.43 € | ~44.8 s | Regularización avanzada y rápida ejecución |
| **LightGBM (Seleccionado)** | **1,492.74 €** | **~25.8 s** | **Mejor precisión y máxima eficiencia** |

### ⚙️ Configuración del Modelo Ganador (`LightGBM`)
```python
LGBMRegressor(
    random_state=12,
    n_estimators=2000,
    max_depth=20,
    min_samples_leaf=5,
    learning_rate=0.1
)
