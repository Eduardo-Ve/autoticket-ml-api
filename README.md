# AutoTicket ML API 
Este repositorio es la **API de Inferencia de ML** para el sistema de autoticket. Desarrollada con **FastAPI**, funciona como el motor de inteligencia que clasifica los tickets de servicio de TI en tiempo real.

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Características

- **Tiempo real:** Expone pipelines de Scikit-Learn entrenados mediante la API REST.
- **Lógica de enrutamiento inteligente:** Aplica umbrales de confianza para distinguir entre asignaciones automatizadas y revisiones manuales.
- **Ligero:** Optimizado para la implementación de CPU mediante TF-IDF y modelos lineales.
- **Seguridad de tipos:** Utiliza Pydantic para una validación estricta de solicitudes/respuestas.

## Estructura de las carpetas

```
autoticket-ml-api/
├── main.py             # API Application entry point
├── models/             # Storage for trained .joblib models
│   ├── category_pipeline.joblib
│   └── meta.json       # Model metadata and thresholds
├── schemas/            # Pydantic data models
└── requirements.txt    # Project dependencies
```
## pre-requsitos
Python 3.11 or higher
Un modelo entrenado colocado en el directorio modelos/.

## instalación

Clona el repositorio:
```
git clone https://github.com/Eduardo-Ve/autoticket-ml-api.git
cd autoticket-ml-api
```
Crea un entorno virtual
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Instala las dependencias que deje en requirements.txt
```
pip install -r requirements.txt

```
Ejecuta el SV
```
uvicorn main:app --reload --port 8000
```
## API Endpoints
1. Health Check
GET / Returns the API status and loaded model version.
2. Classify Ticket
POST /api/classify

Request Body:

```
{
  "text": "My VPN connection keeps dropping every time I try to access the HR portal."
}
```

**Respuesta de ejemplo:**
```

{
  "category": "Access",
  "confidence": 0.85,
  "needs_review": false,
  "probabilities": {
    "Access": 0.85,
    "Hardware": 0.10,
    "Network": 0.05
  }
}
```

## Modelo y logica
El servicio carga una canalización TF-IDF + SGDClassifier. En lugar de predicciones sin procesar, implementa una capa de umbral de confianza

* Ejemplificación:
  * Si prediction_confidence > umbral: Devuelve la categoría específica.
  * Si prediction_confidence < umbral: Devuelve la categoría REVIEW para marcar la intervención humana.

## Proximas metas:
* mejorar las predicciones del modelo
* Implementar el idioma Español para los tickets
* Agregar el notebook de trabajo como referencia
