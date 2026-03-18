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
from flask import Flask, request, jsonify, render_template_string
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


@app.route('/test', methods=['GET'])
def test_ui():
    """Interfaz web para probar la API desde el navegador."""
    html = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>PoliSistemas — Test API</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1 { color: #2c3e50; }
    label { display: block; margin-top: 12px; font-weight: bold; color: #555; }
    input, select { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
    button { margin-top: 20px; width: 100%; padding: 12px; background: #2980b9; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
    button:hover { background: #1a5276; }
    #resultado { margin-top: 20px; padding: 16px; background: white; border-radius: 8px; border-left: 4px solid #2980b9; display: none; }
    #precio { font-size: 2em; color: #27ae60; font-weight: bold; }
    #error { color: #e74c3c; }
  </style>
</head>
<body>
  <h1>Prediccion de Precio de Alquiler</h1>
  <p>Ecuador &mdash; PoliSistemas</p>

  <label>Provincia</label>
  <select id="provincia">
    <option>Pichincha</option><option>Guayas</option><option>El Oro</option>
    <option>Imbabura</option><option>Cotopaxi</option><option>Esmeraldas</option>
    <option>Los Rios</option><option>Manabi</option><option>Orellana</option><option>Santa Elena</option>
  </select>

  <label>Lugar (ciudad)</label>
  <input type="text" id="lugar" value="Quito" placeholder="Ej: Quito, Guayaquil, Machala">

  <label>Numero de dormitorios</label>
  <input type="number" id="dormitorios" value="3" min="0" max="10">

  <label>Numero de banos</label>
  <input type="number" id="banos" value="2" min="0" max="10">

  <label>Area (m2)</label>
  <input type="number" id="area" value="120" min="1">

  <label>Numero de garajes</label>
  <input type="number" id="garages" value="1" min="0" max="10">

  <button onclick="predecir()">Predecir Precio</button>

  <div id="resultado">
    <p>Precio estimado de alquiler:</p>
    <div id="precio"></div>
    <div id="error"></div>
  </div>

  <script>
    async function predecir() {
      const payload = {
        provincia: document.getElementById('provincia').value,
        lugar: document.getElementById('lugar').value,
        num_dormitorios: parseFloat(document.getElementById('dormitorios').value),
        num_banos: parseFloat(document.getElementById('banos').value),
        area: parseFloat(document.getElementById('area').value),
        num_garages: parseFloat(document.getElementById('garages').value)
      };
      const res = document.getElementById('resultado');
      const precioEl = document.getElementById('precio');
      const errorEl = document.getElementById('error');
      res.style.display = 'block';
      precioEl.textContent = 'Calculando...';
      errorEl.textContent = '';
      try {
        const r = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await r.json();
        if (data.prediction !== undefined) {
          precioEl.textContent = '$' + data.prediction.toFixed(2) + ' USD/mes';
          errorEl.textContent = '';
        } else {
          precioEl.textContent = '';
          errorEl.textContent = 'Error: ' + (data.detalle || data.error);
        }
      } catch(e) {
        precioEl.textContent = '';
        errorEl.textContent = 'Error de conexion: ' + e.message;
      }
    }
  </script>
</body>
</html>
"""
    return render_template_string(html)


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
