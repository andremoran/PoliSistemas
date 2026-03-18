# 🏠 Predicción de Precio de Alquiler — Ecuador

**PoliSistemas · Laboratorio de Ciencia de Datos**
Proceso de Selección — Técnico de Investigación

---

## Descripción de la Solución

Flujo completo de ciencia de datos para predecir el precio mensual de alquiler de inmuebles en Ecuador, desde el análisis exploratorio hasta el despliegue de una API REST pública.

### Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| ML | scikit-learn (Gradient Boosting) |
| API | Flask 3 + Gunicorn |
| Deployment | Render |
| Notebooks | Jupyter |

### Estructura del Repositorio

```
├── data/
│   ├── real_state_ecuador_dataset.csv   # Dataset original
│   └── dataset_limpio.csv               # Dataset procesado (generado)
├── model/
│   ├── modelo_precio.pkl                # Modelo serializado (generado)
│   └── metadata.json                    # Métricas y metadata (generado)
├── 1_EDA.ipynb                          # Notebook: Análisis Exploratorio
├── 2_Modelado.ipynb                     # Notebook: Modelado ML
├── app.py                               # API REST (Flask)
├── train_model.py                       # Script de entrenamiento
├── requirements.txt                     # Dependencias
├── Procfile                             # Configuración Render/Heroku
├── render.yaml                          # Configuración declarativa Render
└── README.md
```

---

## API REST — Predicción de Precio

### URL Pública

```
https://polisistemas-rental-api.onrender.com
```

### Endpoints

#### `GET /`  — Health Check

```bash
curl https://polisistemas-rental-api.onrender.com/
```

Respuesta:
```json
{
  "status": "ok",
  "service": "Predicción de Precio de Alquiler — Ecuador",
  "modelo_cargado": true,
  "version": "1.0.0"
}
```

---

#### `GET /info`  — Información del Modelo

```bash
curl https://polisistemas-rental-api.onrender.com/info
```

Respuesta:
```json
{
  "modelo": "GradientBoostingRegressor",
  "metricas": {
    "MAE": 185.34,
    "RMSE": 367.12,
    "R2": 0.7823
  },
  "features": {
    "categoricos": ["provincia", "lugar"],
    "numericos": ["num_dormitorios", "num_banos", "area", "num_garages"]
  }
}
```

---

#### `POST /predict`  — Predicción de Precio

**Entrada:**

```json
{
  "provincia": "Pichincha",
  "lugar": "Quito",
  "num_dormitorios": 3,
  "num_banos": 2,
  "area": 120,
  "num_garages": 1
}
```

**Salida:**

```json
{
  "prediction": 750.0
}
```

### Ejemplo con curl

```bash
curl -X POST https://polisistemas-rental-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "provincia": "Pichincha",
    "lugar": "Quito",
    "num_dormitorios": 3,
    "num_banos": 2,
    "area": 120,
    "num_garages": 1
  }'
```

### Ejemplo con Python

```python
import requests

response = requests.post(
    "https://polisistemas-rental-api.onrender.com/predict",
    json={
        "provincia": "Pichincha",
        "lugar": "Quito",
        "num_dormitorios": 3,
        "num_banos": 2,
        "area": 120,
        "num_garages": 1
    }
)
print(response.json())  # {'prediction': 750.0}
```

### Ejemplo con Postman

1. Método: `POST`
2. URL: `https://polisistemas-rental-api.onrender.com/predict`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
```json
{
  "provincia": "Guayas",
  "lugar": "Guayaquil",
  "num_dormitorios": 2,
  "num_banos": 1,
  "area": 70,
  "num_garages": 0
}
```

### Provincias soportadas

`Pichincha`, `Guayas`, `El Oro`, `Imbabura`, `Cotopaxi`, `Esmeraldas`, `Los Rios`, `Manabí`, `Orellana`, `Santa Elena`

> Para otras provincias no vistas en entrenamiento, el modelo asignará la categoría "desconocida" con valor -1.

---

## Instalación y Ejecución Local

### 1. Clonar repositorio

```bash
git clone https://github.com/<tu-usuario>/polisistemas-rental
cd polisistemas-rental
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Entrenar el modelo

```bash
python train_model.py
```

Esto genera `model/modelo_precio.pkl` y `model/metadata.json`.

### 5. Ejecutar la API localmente

```bash
python app.py
```

La API estará disponible en `http://localhost:5000`.

### 6. Ejecutar los notebooks

```bash
jupyter notebook
```

Abrir en orden:
1. `1_EDA.ipynb` — Análisis exploratorio
2. `2_Modelado.ipynb` — Modelado y evaluación

---

## Modelo de Machine Learning

### Algoritmo: Gradient Boosting Regressor

**Justificación:**
- La distribución de precios tiene cola derecha pronunciada — modelos lineales tienen mal ajuste.
- Manejo nativo de no-linealidades (interacción área × ubicación).
- Robusto a outliers moderados sin necesidad de transformaciones adicionales.
- Supera a Random Forest en datasets de tamaño mediano con hiperparámetros apropiados.

**Hiperparámetros principales:**

| Parámetro | Valor |
|---|---|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 5 |
| `subsample` | 0.8 |
| `min_samples_leaf` | 5 |

**Preprocesamiento:**
- Variables categóricas (`provincia`, `lugar`): `OrdinalEncoder` con manejo de categorías desconocidas.
- Variables numéricas: `StandardScaler`.
- Todo encapsulado en un `sklearn.Pipeline` para garantizar consistencia train/predict.

---

## Despliegue en Render

El archivo `render.yaml` configura el despliegue automático:

1. **Build:** `pip install -r requirements.txt && python train_model.py`
2. **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`

El modelo se entrena automáticamente en cada deploy con los datos del repositorio.

---

## Notebooks

| Notebook | Descripción |
|---|---|
| `1_EDA.ipynb` | Limpieza de datos, normalización de `Lugar`, análisis descriptivo, correlaciones, premium por habitación, clasificación Económico/Medio/Lujo |
| `2_Modelado.ipynb` | Comparación de 4 modelos, evaluación con MAE/RMSE/R², análisis de residuos, importancia de variables, serialización |
