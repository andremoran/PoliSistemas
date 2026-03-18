"""
API REST — Predicción de Precio de Alquiler en Ecuador
======================================================
PoliSistemas · Proceso de Selección Técnico de Investigación

Endpoints:
  GET  /          — Health check
  GET  /info      — Información del modelo
  POST /predict   — Predicción de precio
"""

import os
import json
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Configuración de la aplicación
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['JSON_ENSURE_ASCII'] = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'modelo_precio.pkl')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'model', 'metadata.json')

# Cargar modelo al iniciar la aplicación
try:
    modelo = joblib.load(MODEL_PATH)
    print(f'[OK] Modelo cargado desde: {MODEL_PATH}')
except FileNotFoundError:
    modelo = None
    print(f'[WARNING] Modelo no encontrado en: {MODEL_PATH}')
    print('         Ejecute primero el notebook 2_Modelado.ipynb para generar el modelo.')

# Cargar metadata si existe
try:
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
except FileNotFoundError:
    metadata = {'modelo': 'desconocido', 'features': [], 'metricas': {}}

# Features requeridos para la predicción
FEATURES_REQUERIDOS = ['provincia', 'lugar', 'num_dormitorios', 'num_banos', 'area', 'num_garages']
FEATURES_NUMERICOS = ['num_dormitorios', 'num_banos', 'area', 'num_garages']
FEATURES_CATEGORICOS = ['provincia', 'lugar']


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def validar_entrada(datos: dict) -> tuple[bool, str]:
    """
    Valida que la entrada JSON contenga todos los campos requeridos
    y que los tipos sean correctos.

    Retorna: (es_valido, mensaje_error)
    """
    # Verificar campos faltantes
    faltantes = [f for f in FEATURES_REQUERIDOS if f not in datos]
    if faltantes:
        return False, f"Campos faltantes: {', '.join(faltantes)}"

    # Validar tipos numéricos
    for col in FEATURES_NUMERICOS:
        try:
            val = float(datos[col])
            if val < 0:
                return False, f"El campo '{col}' no puede ser negativo."
        except (ValueError, TypeError):
            return False, f"El campo '{col}' debe ser un número. Recibido: {datos[col]}"

    # Validar categóricos no vacíos
    for col in FEATURES_CATEGORICOS:
        if not str(datos[col]).strip():
            return False, f"El campo '{col}' no puede estar vacío."

    # Validaciones específicas de dominio
    if float(datos['area']) == 0:
        return False, "El 'area' debe ser mayor a 0."

    return True, ""


def preparar_dataframe(datos: dict) -> pd.DataFrame:
    """Convierte el JSON de entrada en un DataFrame compatible con el modelo."""
    return pd.DataFrame([{
        'provincia':       str(datos['provincia']).strip(),
        'lugar':           str(datos['lugar']).strip(),
        'num_dormitorios': float(datos['num_dormitorios']),
        'num_banos':       float(datos['num_banos']),
        'area':            float(datos['area']),
        'num_garages':     float(datos['num_garages'])
    }])


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def health_check():
    """
    Health check de la API.

    Respuesta:
        200 OK con estado del servicio
    """
    return jsonify({
        'status': 'ok',
        'service': 'Predicción de Precio de Alquiler — Ecuador',
        'modelo_cargado': modelo is not None,
        'version': '1.0.0'
    }), 200


@app.route('/info', methods=['GET'])
def info():
    """
    Información del modelo y la API.

    Respuesta:
        200 OK con metadata del modelo y esquema de la API
    """
    return jsonify({
        'modelo': metadata.get('modelo', 'N/A'),
        'metricas': metadata.get('metricas', {}),
        'features': {
            'categoricos': FEATURES_CATEGORICOS,
            'numericos': FEATURES_NUMERICOS
        },
        'ejemplo_request': {
            'provincia': 'Pichincha',
            'lugar': 'Quito',
            'num_dormitorios': 3,
            'num_banos': 2,
            'area': 120,
            'num_garages': 1
        },
        'ejemplo_response': {
            'prediction': 750.0
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicción del precio de alquiler.

    Entrada (JSON):
    {
        "provincia": "Pichincha",
        "lugar": "Quito",
        "num_dormitorios": 3,
        "num_banos": 2,
        "area": 120,
        "num_garages": 1
    }

    Salida (JSON):
    {
        "prediction": 750.0
    }

    Códigos de estado:
        200 — Predicción exitosa
        400 — Datos de entrada inválidos
        503 — Modelo no disponible
    """
    # Verificar que el modelo esté cargado
    if modelo is None:
        return jsonify({
            'error': 'Modelo no disponible.',
            'detalle': 'Ejecute el notebook 2_Modelado.ipynb para generar el modelo.'
        }), 503

    # Parsear JSON de entrada
    datos = request.get_json(silent=True)
    if datos is None:
        return jsonify({
            'error': 'Cuerpo de la solicitud inválido.',
            'detalle': 'Se esperaba un JSON válido con Content-Type: application/json'
        }), 400

    # Validar campos
    es_valido, mensaje_error = validar_entrada(datos)
    if not es_valido:
        return jsonify({
            'error': 'Datos de entrada inválidos.',
            'detalle': mensaje_error
        }), 400

    # Realizar predicción
    try:
        X_input = preparar_dataframe(datos)
        precio_predicho = float(modelo.predict(X_input)[0])
        # Redondear a 2 decimales y asegurar valor positivo
        precio_predicho = max(0.0, round(precio_predicho, 2))

        return jsonify({'prediction': precio_predicho}), 200

    except Exception as e:
        return jsonify({
            'error': 'Error interno al realizar la predicción.',
            'detalle': str(e)
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint no encontrado.', 'endpoints_disponibles': ['GET /', 'GET /info', 'POST /predict']}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Método HTTP no permitido para este endpoint.'}), 405


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    print(f'Iniciando API en puerto {port} (debug={debug})')
    app.run(host='0.0.0.0', port=port, debug=debug)
