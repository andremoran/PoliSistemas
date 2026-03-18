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
@app.route('/ping', methods=['GET'])
def ping():
    """Endpoint minimo para keep-alive (UptimeRobot u otro monitor)."""
    return 'pong', 200


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
    """Interfaz web interactiva con particulas 3D y formulario reactivo."""
    html = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PoliSistemas &mdash; Predicci&oacute;n de Alquiler</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #050a18;
    color: #e0e8ff;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }

  canvas#bg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0;
  }

  .card {
    position: relative;
    z-index: 10;
    background: rgba(10, 20, 50, 0.72);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(100, 160, 255, 0.18);
    border-radius: 20px;
    padding: 36px 44px 28px;
    width: 100%;
    max-width: 490px;
    box-shadow: 0 8px 60px rgba(0, 80, 200, 0.25), 0 0 0 1px rgba(100,160,255,0.08);
    transition: box-shadow 0.4s ease;
  }
  .card:hover {
    box-shadow: 0 12px 80px rgba(0, 120, 255, 0.35), 0 0 0 1px rgba(100,180,255,0.18);
  }

  .header { text-align: center; margin-bottom: 24px; }
  .header .badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a4aff22, #00d4ff22);
    border: 1px solid #1a6aff55;
    color: #7ab8ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 8px;
  }
  .header h1 {
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(135deg, #7ab8ff, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
  }
  .header p { color: #6b85b5; font-size: 13px; }
  .header .stats {
    margin-top: 6px;
    font-size: 11px;
    color: #4a7ab5;
    letter-spacing: 0.5px;
  }
  .header .stats span { color: #4a9eff; font-weight: 700; }

  /* ── Mode toggle ── */
  .mode-toggle {
    display: flex;
    gap: 0;
    border: 1px solid rgba(100,160,255,0.2);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 18px;
  }
  .mode-toggle button {
    flex: 1;
    padding: 8px 0;
    background: transparent;
    border: none;
    color: #5a80c0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
  }
  .mode-toggle button.active {
    background: rgba(74,158,255,0.18);
    color: #7ab8ff;
  }
  .mode-toggle button:first-child { border-right: 1px solid rgba(100,160,255,0.15); }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .full { grid-column: 1 / -1; }

  .field { display: flex; flex-direction: column; gap: 5px; }

  label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #5a80c0;
    transition: color 0.25s;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .field:focus-within label { color: #7ab8ff; }
  .lval {
    color: #4a9eff;
    font-size: 13px;
    text-transform: none;
    letter-spacing: 0;
    font-weight: 700;
  }

  input, select {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,160,255,0.15);
    border-radius: 10px;
    padding: 10px 14px;
    color: #e0e8ff;
    font-size: 14px;
    outline: none;
    transition: border-color 0.25s, background 0.25s, box-shadow 0.25s, transform 0.15s;
    cursor: pointer;
    width: 100%;
    appearance: none;
    -webkit-appearance: none;
  }
  select { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%235a80c0' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 12px center; padding-right: 32px; }
  select option { background: #0d1b3e; color: #e0e8ff; }

  input:focus, select:focus {
    border-color: rgba(100,180,255,0.5);
    background: rgba(100,160,255,0.08);
    box-shadow: 0 0 0 3px rgba(100,160,255,0.1), 0 0 20px rgba(100,160,255,0.08);
    transform: translateY(-1px);
  }
  input:hover, select:hover {
    border-color: rgba(100,160,255,0.3);
    background: rgba(255,255,255,0.06);
  }

  input[type="number"]::-webkit-inner-spin-button { opacity: 0.4; filter: invert(1); }

  input[type="range"] {
    -webkit-appearance: auto;
    appearance: auto;
    padding: 0;
    height: 4px;
    cursor: pointer;
    accent-color: #4a9eff;
    background: rgba(100,160,255,0.15);
    border: none;
    border-radius: 4px;
    transform: none;
  }
  input[type="range"]:focus { box-shadow: none; transform: none; }

  /* ── JSON mode ── */
  #json-mode { display: none; }
  #json-input {
    width: 100%;
    height: 140px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(100,160,255,0.2);
    border-radius: 10px;
    color: #a0d4ff;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 12px;
    resize: vertical;
    outline: none;
    transition: border-color 0.25s;
  }
  #json-input:focus { border-color: rgba(100,180,255,0.5); }
  .json-actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }
  .btn-sm {
    flex: 1;
    padding: 8px 0;
    border: 1px solid rgba(100,160,255,0.25);
    border-radius: 8px;
    background: rgba(74,158,255,0.08);
    color: #7ab8ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s;
  }
  .btn-sm:hover { background: rgba(74,158,255,0.18); border-color: rgba(100,160,255,0.5); }
  .btn-upload {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    width: 100%;
    padding: 9px 0;
    margin-top: 8px;
    border: 1px dashed rgba(100,160,255,0.3);
    border-radius: 8px;
    background: rgba(74,158,255,0.04);
    color: #4a7ab5;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s, color 0.2s;
  }
  .btn-upload:hover { background: rgba(74,158,255,0.12); border-color: rgba(100,160,255,0.55); color: #7ab8ff; }
  #file-upload { display: none; }

  #json-response-wrap { display: none; margin-top: 10px; }
  #json-response-wrap label { margin-bottom: 4px; display: block; color: #4a7ab5; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
  #json-response {
    width: 100%;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(100,160,255,0.15);
    border-radius: 10px;
    color: #4aff8c;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 10px 12px;
    min-height: 48px;
    white-space: pre;
  }

  .btn {
    width: 100%;
    padding: 14px;
    margin-top: 8px;
    border: none;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.5px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #1a5fff, #00c8ff);
    color: white;
    transition: transform 0.15s, box-shadow 0.25s, filter 0.25s;
    box-shadow: 0 4px 24px rgba(26,95,255,0.35);
  }
  .btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(26,95,255,0.5);
    filter: brightness(1.1);
  }
  .btn:active { transform: translateY(0); }
  .btn::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at var(--mx,50%) var(--my,50%), rgba(255,255,255,0.25) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
  }
  .btn:hover::after { opacity: 1; }
  .btn.loading { pointer-events: none; filter: brightness(0.8); }

  #resultado {
    margin-top: 20px;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(100,160,255,0.15);
    background: rgba(10,40,100,0.35);
    display: none;
    text-align: center;
    animation: fadeUp 0.4s cubic-bezier(0.22,1,0.36,1);
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .result-label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #5a80c0; margin-bottom: 6px; }
  #precio {
    font-size: 2.6em;
    font-weight: 800;
    background: linear-gradient(135deg, #00e5ff, #4aff8c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
  }
  .result-sub { font-size: 12px; color: #4a6a9a; margin-top: 4px; }
  #error-msg { color: #ff6b6b; font-size: 13px; }

  .dots-loader {
    display: inline-flex; gap: 5px; align-items: center; justify-content: center;
    height: 24px;
  }
  .dots-loader span {
    width: 6px; height: 6px; border-radius: 50%; background: #4a9eff;
    animation: bounce 0.9s infinite ease-in-out;
  }
  .dots-loader span:nth-child(2) { animation-delay: 0.15s; }
  .dots-loader span:nth-child(3) { animation-delay: 0.30s; }
  @keyframes bounce {
    0%,80%,100% { transform: scale(0.6); opacity: 0.4; }
    40%          { transform: scale(1.1); opacity: 1; }
  }

  /* ── Footer ── */
  .footer {
    margin-top: 18px;
    text-align: center;
    font-size: 11px;
    color: #2a4a7a;
    line-height: 1.8;
  }
  .footer a { color: #3a6aaa; text-decoration: none; }
  .footer a:hover { color: #7ab8ff; }
</style>
</head>
<body>

<canvas id="bg"></canvas>

<div class="card" id="card">
  <div class="header">
    <div class="badge">PoliSistemas &middot; ML</div>
    <h1>Predicci&oacute;n de Alquiler</h1>
    <p>Ecuador &mdash; Gradient Boosting Regressor</p>
    <div class="stats">R&sup2; <span>0.62</span> &nbsp;&bull;&nbsp; <span>483</span> inmuebles reales &nbsp;&bull;&nbsp; MAE &lt; $185</div>
  </div>

  <div class="mode-toggle">
    <button id="btn-visual" class="active" onclick="setMode('visual')">&#9776; Modo Visual</button>
    <button id="btn-json"   onclick="setMode('json')">&#123;&#125; Modo JSON</button>
  </div>

  <!-- ── Modo Visual ── -->
  <div id="visual-mode">
    <div class="grid">
      <div class="field full">
        <label>Provincia</label>
        <select id="provincia">
          <option>Pichincha</option>
          <option>Guayas</option>
          <option>El Oro</option>
          <option>Imbabura</option>
          <option>Cotopaxi</option>
          <option>Esmeraldas</option>
          <option>Los Rios</option>
          <option>Manabi</option>
          <option>Orellana</option>
          <option>Santa Elena</option>
        </select>
      </div>

      <div class="field full">
        <label>Ciudad / Lugar</label>
        <input type="text" id="lugar" value="Quito" placeholder="Quito, Guayaquil, Machala...">
      </div>

      <div class="field">
        <label>Dormitorios <span class="lval" id="vDorm">3</span></label>
        <input type="range" id="dormitorios" min="1" max="6" value="3" oninput="updateVal('dormitorios','vDorm')">
      </div>

      <div class="field">
        <label>Ba&ntilde;os <span class="lval" id="vBano">2</span></label>
        <input type="range" id="banos" min="1" max="5" value="2" oninput="updateVal('banos','vBano')">
      </div>

      <div class="field full">
        <label>Area (m&sup2;) <span class="lval" id="vArea">120</span></label>
        <input type="range" id="area" min="20" max="500" value="120" oninput="updateVal('area','vArea')">
      </div>

      <div class="field full">
        <label>Garajes <span class="lval" id="vGar">1</span></label>
        <input type="range" id="garages" min="0" max="4" value="1" oninput="updateVal('garages','vGar')">
      </div>
    </div>
  </div>

  <!-- ── Modo JSON ── -->
  <div id="json-mode">
    <textarea id="json-input" spellcheck="false">{
  "provincia": "Pichincha",
  "lugar": "Quito",
  "num_dormitorios": 3,
  "num_banos": 2,
  "area": 120,
  "num_garages": 1
}</textarea>
    <div class="json-actions">
      <button class="btn-sm" onclick="formatJson()">Formatear JSON</button>
      <button class="btn-sm" onclick="syncFromSliders()">Sync desde sliders</button>
    </div>
    <label class="btn-upload" for="file-upload">&#8593; Subir archivo .json</label>
    <input type="file" id="file-upload" accept=".json,application/json" onchange="loadJsonFile(event)">
    <div id="json-response-wrap">
      <label>Respuesta</label>
      <div id="json-response"></div>
    </div>
  </div>

  <button class="btn" id="btnPredecir" onclick="predecir()">Predecir Precio</button>

  <div id="resultado">
    <div class="result-label">Precio estimado de alquiler</div>
    <div id="precio"></div>
    <div class="result-sub" id="result-sub"></div>
    <div id="error-msg"></div>
  </div>

  <div class="footer">
    API: <a href="/predict" target="_blank">/predict</a> &nbsp;&bull;&nbsp;
    <a href="/info" target="_blank">/info</a> &nbsp;&bull;&nbsp;
    <a href="https://github.com/andremoran/PoliSistemas" target="_blank">GitHub</a>
  </div>
</div>

<script>
// ── Mode toggle ──────────────────────────────────────────────────────────────
let currentMode = 'visual';
function setMode(mode) {
  currentMode = mode;
  document.getElementById('visual-mode').style.display = mode === 'visual' ? 'block' : 'none';
  document.getElementById('json-mode').style.display   = mode === 'json'   ? 'block' : 'none';
  document.getElementById('btn-visual').classList.toggle('active', mode === 'visual');
  document.getElementById('btn-json').classList.toggle('active', mode === 'json');
  if (mode === 'json') syncFromSliders();
}

function syncFromSliders() {
  const payload = getPayload();
  document.getElementById('json-input').value = JSON.stringify(payload, null, 2);
}

function loadJsonFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    try {
      const parsed = JSON.parse(e.target.result);
      document.getElementById('json-input').value = JSON.stringify(parsed, null, 2);
      // Resetear el input para que se pueda subir el mismo archivo de nuevo
      event.target.value = '';
    } catch(err) {
      document.getElementById('json-input').value = e.target.result;
      event.target.value = '';
    }
  };
  reader.readAsText(file);
}

function formatJson() {
  try {
    const parsed = JSON.parse(document.getElementById('json-input').value);
    document.getElementById('json-input').value = JSON.stringify(parsed, null, 2);
  } catch(e) {
    document.getElementById('json-input').style.borderColor = 'rgba(255,100,100,0.5)';
    setTimeout(() => document.getElementById('json-input').style.borderColor = '', 1200);
  }
}

// ── Sliders ──────────────────────────────────────────────────────────────────
function updateVal(id, valId) {
  document.getElementById(valId).textContent = document.getElementById(id).value;
}

// ── Payload builder ──────────────────────────────────────────────────────────
function getPayload() {
  if (currentMode === 'json') {
    try { return JSON.parse(document.getElementById('json-input').value); }
    catch(e) { throw new Error('JSON inválido: ' + e.message); }
  }
  return {
    provincia:       document.getElementById('provincia').value,
    lugar:           document.getElementById('lugar').value.trim() || 'Quito',
    num_dormitorios: parseFloat(document.getElementById('dormitorios').value),
    num_banos:       parseFloat(document.getElementById('banos').value),
    area:            parseFloat(document.getElementById('area').value),
    num_garages:     parseFloat(document.getElementById('garages').value)
  };
}

// ── Boton ripple ─────────────────────────────────────────────────────────────
document.getElementById('btnPredecir').addEventListener('mousemove', function(e) {
  const r = this.getBoundingClientRect();
  const x = ((e.clientX - r.left) / r.width * 100).toFixed(1);
  const y = ((e.clientY - r.top)  / r.height * 100).toFixed(1);
  this.style.setProperty('--mx', x + '%');
  this.style.setProperty('--my', y + '%');
});

// ── Prediccion ───────────────────────────────────────────────────────────────
async function predecir() {
  const btn    = document.getElementById('btnPredecir');
  const res    = document.getElementById('resultado');
  const precioEl = document.getElementById('precio');
  const subEl  = document.getElementById('result-sub');
  const errEl  = document.getElementById('error-msg');

  let payload;
  try { payload = getPayload(); }
  catch(e) {
    res.style.display = 'block';
    precioEl.textContent = '';
    errEl.textContent = e.message;
    return;
  }

  btn.classList.add('loading');
  btn.innerHTML = '<div class="dots-loader"><span></span><span></span><span></span></div>';
  res.style.display = 'block';
  precioEl.innerHTML = '<div class="dots-loader"><span></span><span></span><span></span></div>';
  subEl.textContent = '';
  errEl.textContent = '';

  // Ocultar respuesta JSON anterior
  document.getElementById('json-response-wrap').style.display = 'none';

  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await r.json();

    // En modo JSON mostrar respuesta en textarea
    if (currentMode === 'json') {
      const wrap = document.getElementById('json-response-wrap');
      document.getElementById('json-response').textContent = JSON.stringify(data, null, 2);
      wrap.style.display = 'block';
    }

    if (data.prediction !== undefined) {
      const p = data.prediction;
      animateNumber(precioEl, 0, p, 900);
      subEl.textContent = payload.num_dormitorios + ' dorm \u00b7 ' + payload.num_banos + ' ba\u00f1 \u00b7 ' + payload.area + 'm\u00B2 \u00b7 ' + payload.num_garages + ' gar \u00b7 ' + payload.lugar + ', ' + payload.provincia;
    } else {
      precioEl.textContent = '';
      errEl.textContent = data.detalle || data.error || 'Error desconocido';
    }
  } catch(e) {
    precioEl.textContent = '';
    errEl.textContent = 'Error de conexi\u00f3n: ' + e.message;
  }

  btn.classList.remove('loading');
  btn.textContent = 'Predecir Precio';
}

function animateNumber(el, from, to, duration) {
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 4);
    const val = from + (to - from) * ease;
    el.textContent = '$' + val.toFixed(2) + ' USD/mes';
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Card tilt on mouse ────────────────────────────────────────────────────────
const card = document.getElementById('card');
document.addEventListener('mousemove', e => {
  const cx = window.innerWidth / 2, cy = window.innerHeight / 2;
  const dx = (e.clientX - cx) / cx, dy = (e.clientY - cy) / cy;
  card.style.transform = `perspective(800px) rotateY(${dx * 4}deg) rotateX(${-dy * 3}deg)`;
});
document.addEventListener('mouseleave', () => {
  card.style.transform = 'perspective(800px) rotateY(0deg) rotateX(0deg)';
});

// ── Particulas 3D ────────────────────────────────────────────────────────────
const canvas = document.getElementById('bg');
const ctx    = canvas.getContext('2d');

let W, H, mouse = { x: 0, y: 0 }, particles = [];
const N = 120, FOV = 400, SPEED = 0.4;

function resize() {
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;
}
resize();
window.addEventListener('resize', resize);

document.addEventListener('mousemove', e => {
  mouse.x = e.clientX;
  mouse.y = e.clientY;
});

class Particle {
  constructor() { this.reset(true); }
  reset(init) {
    this.x = (Math.random() - 0.5) * 1400;
    this.y = (Math.random() - 0.5) * 1000;
    this.z = init ? Math.random() * 800 : 800;
    this.vx = (Math.random() - 0.5) * 0.6;
    this.vy = (Math.random() - 0.5) * 0.6;
    this.vz = -SPEED - Math.random() * 0.5;
    this.r  = Math.random() * 1.8 + 0.4;
    this.hue = 190 + Math.random() * 60;
  }
  project() {
    const scale = FOV / (FOV + this.z);
    return { sx: this.x * scale + W / 2, sy: this.y * scale + H / 2, scale };
  }
  update() {
    const { sx, sy } = this.project();
    const mdx = (mouse.x - sx) * 0.0004;
    const mdy = (mouse.y - sy) * 0.0004;
    this.vx += mdx; this.vy += mdy;
    this.vx *= 0.97; this.vy *= 0.97;
    this.x += this.vx; this.y += this.vy; this.z += this.vz;
    if (this.z < -FOV) this.reset(false);
  }
  draw() {
    const { sx, sy, scale } = this.project();
    if (sx < -10 || sx > W + 10 || sy < -10 || sy > H + 10) return;
    const alpha = Math.min(scale * 2, 0.9);
    const size  = this.r * scale * 2.5;
    ctx.beginPath();
    ctx.arc(sx, sy, size, 0, Math.PI * 2);
    ctx.fillStyle = `hsla(${this.hue},80%,70%,${alpha})`;
    ctx.fill();
  }
}

for (let i = 0; i < N; i++) particles.push(new Particle());

function connectParticles() {
  const pts = particles.map(p => p.project());
  const MAX_DIST = 110;
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const dx = pts[i].sx - pts[j].sx;
      const dy = pts[i].sy - pts[j].sy;
      const d  = Math.sqrt(dx*dx + dy*dy);
      if (d < MAX_DIST) {
        const alpha = (1 - d / MAX_DIST) * 0.25 * Math.min(pts[i].scale, pts[j].scale) * 3;
        ctx.beginPath();
        ctx.moveTo(pts[i].sx, pts[i].sy);
        ctx.lineTo(pts[j].sx, pts[j].sy);
        ctx.strokeStyle = `rgba(80,160,255,${alpha})`;
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }
    }
  }
}

function loop() {
  ctx.fillStyle = 'rgba(5,10,24,0.18)';
  ctx.fillRect(0, 0, W, H);
  particles.forEach(p => { p.update(); p.draw(); });
  connectParticles();
  requestAnimationFrame(loop);
}
loop();
</script>
</body>
</html>"""
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
