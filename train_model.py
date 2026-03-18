"""
Script de Entrenamiento del Modelo
===================================
Ejecutar este script para generar el archivo model/modelo_precio.pkl
a partir del dataset crudo.

Uso:
    python train_model.py

Requisitos:
    pip install -r requirements.txt
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'real_state_ecuador_dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_precio.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')


def limpiar_lugar(lugar_raw: str, provincia: str) -> str:
    """Extrae el nombre de ciudad desde una dirección completa."""
    if pd.isna(lugar_raw):
        return provincia
    partes = [p.strip() for p in str(lugar_raw).split(',')]
    partes_limpias = []
    for p in partes:
        if not p or p.lower() == 'ecuador':
            continue
        if re.match(r'^[A-Z0-9]{4}\+[A-Z0-9]+', p) or re.match(r'^\d+$', p):
            continue
        partes_limpias.append(p)
    if not partes_limpias:
        return provincia
    if partes_limpias[0].lower().replace(' ', '') == provincia.lower().replace(' ', ''):
        partes_limpias = partes_limpias[1:]
    if not partes_limpias:
        return provincia
    ciudad = re.sub(r'\s+\d{5,6}\b', '', partes_limpias[-1]).strip()
    return ciudad if ciudad else provincia


def cargar_y_limpiar_datos(path: str) -> pd.DataFrame:
    """Carga el CSV crudo y aplica limpieza básica."""
    df = pd.read_csv(path)
    df.columns = ['titulo', 'precio', 'provincia', 'lugar_raw', 'num_dormitorios', 'num_banos', 'area', 'num_garages']

    # Convertir numéricos y manejar NaN
    cols_num = ['num_dormitorios', 'num_banos', 'area', 'num_garages', 'precio']
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Imputar numéricos con mediana por provincia
    for col in ['num_dormitorios', 'num_banos', 'area', 'num_garages']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(
                df.groupby('provincia')[col].transform('median')
            ).fillna(df[col].median())

    # Eliminar filas sin precio (target no puede ser NaN)
    df = df.dropna(subset=['precio'])

    # Normalizar columna Lugar
    df['lugar'] = df.apply(lambda r: limpiar_lugar(r['lugar_raw'], r['provincia']), axis=1)

    return df


def construir_pipeline() -> Pipeline:
    """
    Construye el pipeline de preprocesamiento + modelo.

    Se aplica transformacion log1p al target (precio) para manejar la
    distribucion sesgada y reducir el impacto de outliers extremos.
    La prediccion final se convierte de vuelta con expm1.
    """
    cols_cat = ['provincia', 'lugar']
    cols_num = ['num_dormitorios', 'num_banos', 'area', 'num_garages']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cols_cat),
            ('num', StandardScaler(), cols_num)
        ],
        remainder='drop'
    )

    gbm = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )

    # TransformedTargetRegressor aplica log1p al target antes de entrenar
    # y expm1 al predecir, haciendo transparente la transformacion
    regresor_con_transform = TransformedTargetRegressor(
        regressor=gbm,
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', regresor_con_transform)
    ])

    return pipeline


def entrenar_y_evaluar(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Entrena el modelo y calcula métricas de evaluación."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2').mean()

    metricas = {
        'MAE': round(float(mae), 2),
        'RMSE': round(float(rmse), 2),
        'R2': round(float(r2), 4),
        'CV_R2_5fold': round(float(cv_r2), 4),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    }

    return metricas


def main():
    print('=' * 55)
    print('  Entrenamiento del Modelo - PoliSistemas')
    print('=' * 55)

    # 1. Cargar datos
    print(f'\n[1/4] Cargando dataset desde: {DATA_PATH}')
    df = cargar_y_limpiar_datos(DATA_PATH)
    print(f'      Registros cargados: {len(df)}')

    # 2. Preparar features
    FEATURES = ['provincia', 'lugar', 'num_dormitorios', 'num_banos', 'area', 'num_garages']
    TARGET = 'precio'
    X = df[FEATURES]
    y = df[TARGET]
    print(f'\n[2/4] Features: {FEATURES}')
    print(f'      Target: {TARGET} | Rango: ${y.min():.0f} – ${y.max():.0f}')

    # 3. Construir y evaluar pipeline
    print('\n[3/4] Entrenando Gradient Boosting Regressor...')
    pipeline = construir_pipeline()
    # Re-entrenar en todos los datos después de evaluar
    metricas = entrenar_y_evaluar(construir_pipeline(), X, y)
    pipeline.fit(X, y)  # Modelo final entrenado en 100% de los datos

    print(f'      MAE:         ${metricas["MAE"]:.2f}')
    print(f'      RMSE:        ${metricas["RMSE"]:.2f}')
    print(f'      R²:          {metricas["R2"]:.4f}')
    print(f'      CV R²:       {metricas["CV_R2_5fold"]:.4f} (5-fold)')

    # 4. Guardar modelo y metadata
    print(f'\n[4/4] Guardando modelo en: {MODEL_PATH}')
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        'modelo': 'GradientBoostingRegressor',
        'features': FEATURES,
        'target': TARGET,
        'metricas': metricas
    }
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f'      Metadata guardada en: {METADATA_PATH}')

    # Prueba rápida
    ejemplo = pd.DataFrame([{
        'provincia': 'Pichincha', 'lugar': 'Quito',
        'num_dormitorios': 3, 'num_banos': 2,
        'area': 120, 'num_garages': 1
    }])
    pred_ejemplo = pipeline.predict(ejemplo)[0]
    print(f'\n[OK] Prueba: Quito, 3 dorm, 2 banos, 120m2, 1 garage -> ${pred_ejemplo:.2f}/mes')
    print('\n[OK] Modelo listo para la API.')


if __name__ == '__main__':
    main()
