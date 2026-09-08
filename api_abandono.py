# Web API para predecir el abandono de empleados
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Cargamos el modelo y sus transformaciones
artefactos = joblib.load("modelo_abandono.joblib")
modelo = artefactos["modelo"]
escalador = artefactos["escalador"]
codificadores = artefactos["codificadores"]
columnas = artefactos["columnas"]

app = FastAPI(title="API de predicción de abandono de empleados")


class Empleado(BaseModel):
    edad: int
    viajes: str
    departamento: str
    distancia_casa: int
    educacion: str
    carrera: str
    id: int
    satisfaccion_entorno: str
    sexo: float
    implicacion: str
    nivel_laboral: int
    puesto: str
    satisfaccion_trabajo: str
    estado_civil: str
    salario_mes: int
    num_empresas_anteriores: int
    horas_extra: str
    incremento_salario_porc: int
    evaluacion: str
    satisfaccion_companeros: str
    nivel_acciones: int
    anos_experiencia: int
    num_formaciones_ult_ano: int
    anos_compania: int
    anos_desde_ult_promocion: int
    anos_con_manager_actual: int


@app.get("/")
def raiz():
    return {"mensaje": "API de predicción de abandono de empleados"}


@app.post("/predecir")
def predecir(empleado: Empleado):
    # Convertimos a DataFrame respetando el orden de columnas del modelo
    fila = pd.DataFrame([empleado.model_dump()])[columnas]

    # Codificamos las variables categóricas
    for col in codificadores:
        if col != "abandono":
            fila[col] = codificadores[col].transform(fila[col])

    # Escalamos y predecimos
    fila_escalada = escalador.transform(fila)
    prediccion = modelo.predict(fila_escalada)[0]
    resultado = codificadores["abandono"].inverse_transform([prediccion])[0]

    return {"prediccion_abandono": resultado}
