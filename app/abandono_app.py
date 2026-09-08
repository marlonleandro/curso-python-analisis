"""
Aplicación Streamlit — Predicción de abandono de empleados
==========================================================

Utiliza el modelo entrenado en el notebook 16
(`16.Introduccion_al_Machine_Learning.ipynb`) para predecir si un
empleado abandonará la empresa a partir de sus datos.

El modelo (k-NN), el escalador (`StandardScaler`) y los codificadores
(`LabelEncoder`) se cargan desde `modelo_abandono.joblib`, que es el
archivo generado al final del notebook.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

RUTAS_MODELO = [
    ROOT / "exports" / "modelo_abandono.joblib",
    ROOT / "modelo_abandono.joblib",
]

# ---------------------------------------------------------------------------
# Etiquetas y valores para el formulario
# ---------------------------------------------------------------------------
ETIQUETAS = {
    "edad": "Edad (años)",
    "viajes": "Frecuencia de viajes de trabajo",
    "departamento": "Departamento",
    "distancia_casa": "Distancia al trabajo (km)",
    "educacion": "Nivel educativo",
    "carrera": "Carrera / especialidad",
    "satisfaccion_entorno": "Satisfacción con el entorno",
    "sexo": "Sexo",
    "implicacion": "Nivel de implicación",
    "nivel_laboral": "Nivel laboral",
    "puesto": "Puesto",
    "satisfaccion_trabajo": "Satisfacción con el trabajo",
    "estado_civil": "Estado civil",
    "salario_mes": "Salario mensual",
    "num_empresas_anteriores": "Nº de empresas anteriores",
    "horas_extra": "Realiza horas extra",
    "incremento_salario_porc": "Incremento salarial (%)",
    "evaluacion": "Evaluación de desempeño",
    "satisfaccion_companeros": "Satisfacción con los compañeros",
    "nivel_acciones": "Nivel de acciones en la empresa",
    "anos_experiencia": "Años de experiencia",
    "num_formaciones_ult_ano": "Formaciones en el último año",
    "anos_compania": "Años en la empresa",
    "anos_desde_ult_promocion": "Años desde la última promoción",
    "anos_con_manager_actual": "Años con el manager actual",
}

# Rangos y valores por defecto de las variables numéricas (según el dataset).
NUMERICOS = {
    "edad": {"min_value": 18, "max_value": 60, "value": 34, "step": 1},
    "distancia_casa": {"min_value": 1, "max_value": 29, "value": 4, "step": 1},
    "nivel_laboral": {"min_value": 1, "max_value": 5, "value": 2, "step": 1},
    "salario_mes": {"min_value": 0, "max_value": 30000, "value": 6500, "step": 100},
    "num_empresas_anteriores": {"min_value": 0, "max_value": 9, "value": 3, "step": 1},
    "incremento_salario_porc": {"min_value": 0, "max_value": 30, "value": 12, "step": 1},
    "nivel_acciones": {"min_value": 0, "max_value": 3, "value": 1, "step": 1},
    "anos_experiencia": {"min_value": 0, "max_value": 40, "value": 6, "step": 1},
    "num_formaciones_ult_ano": {"min_value": 0, "max_value": 6, "value": 2, "step": 1},
    "anos_compania": {"min_value": 0, "max_value": 40, "value": 2, "step": 1},
    "anos_desde_ult_promocion": {"min_value": 0, "max_value": 15, "value": 1, "step": 1},
    "anos_con_manager_actual": {"min_value": 0, "max_value": 17, "value": 1, "step": 1},
}

# Texto legible para las categorías (el valor real que recibe el modelo no cambia).
FORMATOS = {
    "viajes": {
        "Non-Travel": "No viaja",
        "Travel_Frequently": "Viaja con frecuencia",
        "Travel_Rarely": "Viaja poco",
    },
    "departamento": {
        "Human Resources": "Recursos Humanos",
        "Research & Development": "Investigación y Desarrollo",
        "Sales": "Ventas",
    },
    "educacion": {
        "Master": "Máster",
        "Primaria": "Primaria",
        "Secundaria": "Secundaria",
        "Universitaria": "Universitaria",
    },
    "carrera": {
        "Human Resources": "Recursos Humanos",
        "Life Sciences": "Ciencias de la Vida",
        "Marketing": "Marketing",
        "Medical": "Medicina",
        "Other": "Otra",
        "Technical Degree": "Título técnico",
    },
    "implicacion": {
        "Alta": "Alta",
        "Baja": "Baja",
        "Media": "Media",
        "Muy_Alta": "Muy alta",
    },
    "puesto": {
        "Healthcare Representative": "Representante sanitario",
        "Human Resources": "Recursos Humanos",
        "Laboratory Technician": "Técnico de laboratorio",
        "Manager": "Gerente",
        "Manufacturing Director": "Director de fabricación",
        "Research Director": "Director de investigación",
        "Research Scientist": "Científico de investigación",
        "Sales Executive": "Ejecutivo de ventas",
        "Sales Representative": "Representante de ventas",
    },
    "satisfaccion_trabajo": {
        "Alta": "Alta",
        "Baja": "Baja",
        "Media": "Media",
        "Muy_Alta": "Muy alta",
    },
    "estado_civil": {
        "Divorced": "Divorciado/a",
        "Married": "Casado/a",
        "Single": "Soltero/a",
    },
    "horas_extra": {"No": "No", "Yes": "Sí"},
    "evaluacion": {"Alta": "Alta", "Muy_Alta": "Muy alta"},
    "satisfaccion_entorno": {
        "Alta": "Alta",
        "Baja": "Baja",
        "Media": "Media",
        "Muy_Alta": "Muy alta",
    },
    "satisfaccion_companeros": {
        "Alta": "Alta",
        "Baja": "Baja",
        "Media": "Media",
        "Muy_Alta": "Muy alta",
    },
    "sexo": {
        "F": "Femenino",
        "M": "Masculino",
    },
}


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo() -> dict:
    """Carga el modelo y sus transformaciones desde disco."""
    for ruta in RUTAS_MODELO:
        if ruta.exists():
            return joblib.load(ruta)
    raise FileNotFoundError(
        "No se encontró el archivo del modelo. Genera 'modelo_abandono.joblib' "
        "ejecutando el notebook 16."
    )


def predecir(datos: dict, artefactos: dict) -> tuple[str, float, float]:
    """Devuelve (resultado, prob_no_abandono, prob_abandono)."""
    modelo = artefactos["modelo"]
    escalador = artefactos["escalador"]
    codificadores = artefactos["codificadores"]
    columnas = artefactos["columnas"]

    # Respetamos el orden de columnas con el que se entrenó el modelo.
    fila = pd.DataFrame([datos])[columnas]

    # Codificamos las variables categóricas con los codificadores guardados.
    for col in codificadores:
        if col != "abandono":
            fila[col] = codificadores[col].transform(fila[col])

    # Escalamos y predecimos.
    fila_escalada = escalador.transform(fila)
    prediccion = modelo.predict(fila_escalada)[0]
    probabilidades = modelo.predict_proba(fila_escalada)[0]

    resultado = codificadores["abandono"].inverse_transform([prediccion])[0]

    # Las clases están en el orden en que las aprendió el LabelEncoder.
    clases = list(codificadores["abandono"].classes_)
    prob_no = float(probabilidades[clases.index("No")])
    prob_yes = float(probabilidades[clases.index("Yes")])

    return resultado, prob_no, prob_yes


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Predicción de abandono de empleados",
        page_icon="🧑‍💼",
        layout="wide",
    )

    st.title("🧑‍💼 Predicción de abandono de empleados")
    st.markdown(
        """
        Completa el formulario con los datos del empleado y obtén una
        **predicción de si abandonará la empresa**, según el modelo entrenado
        en el notebook 16 (clasificador **k-NN**).
        """
    )

    try:
        artefactos = cargar_modelo()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    codificadores = artefactos["codificadores"]
    columnas = artefactos["columnas"]

    with st.form("formulario_empleado"):
        datos: dict = {}

        # --- Datos personales ------------------------------------------------
        st.subheader("👤 Datos personales")
        c1, c2, c3 = st.columns(3)

        with c1:
            datos["edad"] = st.number_input(
                ETIQUETAS["edad"], **NUMERICOS["edad"]
            )
            datos["sexo"] = st.selectbox(
                ETIQUETAS["sexo"],
                list(codificadores["sexo"].classes_),
                format_func=FORMATOS["sexo"].get,
            )
            datos["estado_civil"] = st.selectbox(
                ETIQUETAS["estado_civil"],
                list(codificadores["estado_civil"].classes_),
                format_func=FORMATOS["estado_civil"].get,
            )

        with c2:
            datos["educacion"] = st.selectbox(
                ETIQUETAS["educacion"],
                list(codificadores["educacion"].classes_),
                format_func=FORMATOS["educacion"].get,
            )
            datos["carrera"] = st.selectbox(
                ETIQUETAS["carrera"],
                list(codificadores["carrera"].classes_),
                format_func=FORMATOS["carrera"].get,
            )
            datos["departamento"] = st.selectbox(
                ETIQUETAS["departamento"],
                list(codificadores["departamento"].classes_),
                format_func=FORMATOS["departamento"].get,
            )

        with c3:
            datos["puesto"] = st.selectbox(
                ETIQUETAS["puesto"],
                list(codificadores["puesto"].classes_),
                format_func=FORMATOS["puesto"].get,
            )
            datos["nivel_laboral"] = st.number_input(
                ETIQUETAS["nivel_laboral"], **NUMERICOS["nivel_laboral"]
            )
            datos["salario_mes"] = st.number_input(
                ETIQUETAS["salario_mes"], **NUMERICOS["salario_mes"]
            )

        # --- Trayectoria en la empresa ---------------------------------------
        st.subheader("🏢 Trayectoria laboral")
        c1, c2, c3 = st.columns(3)

        with c1:
            datos["anos_experiencia"] = st.number_input(
                ETIQUETAS["anos_experiencia"], **NUMERICOS["anos_experiencia"]
            )
            datos["anos_compania"] = st.number_input(
                ETIQUETAS["anos_compania"], **NUMERICOS["anos_compania"]
            )
            datos["anos_desde_ult_promocion"] = st.number_input(
                ETIQUETAS["anos_desde_ult_promocion"],
                **NUMERICOS["anos_desde_ult_promocion"],
            )

        with c2:
            datos["anos_con_manager_actual"] = st.number_input(
                ETIQUETAS["anos_con_manager_actual"],
                **NUMERICOS["anos_con_manager_actual"],
            )
            datos["num_empresas_anteriores"] = st.number_input(
                ETIQUETAS["num_empresas_anteriores"],
                **NUMERICOS["num_empresas_anteriores"],
            )
            datos["nivel_acciones"] = st.number_input(
                ETIQUETAS["nivel_acciones"], **NUMERICOS["nivel_acciones"]
            )

        with c3:
            datos["incremento_salario_porc"] = st.number_input(
                ETIQUETAS["incremento_salario_porc"],
                **NUMERICOS["incremento_salario_porc"],
            )
            datos["num_formaciones_ult_ano"] = st.number_input(
                ETIQUETAS["num_formaciones_ult_ano"],
                **NUMERICOS["num_formaciones_ult_ano"],
            )
            datos["distancia_casa"] = st.number_input(
                ETIQUETAS["distancia_casa"], **NUMERICOS["distancia_casa"]
            )

        # --- Condiciones y satisfacción ---------------------------------------
        st.subheader("😊 Condiciones y satisfacción")
        c1, c2, c3 = st.columns(3)

        with c1:
            datos["viajes"] = st.selectbox(
                ETIQUETAS["viajes"],
                list(codificadores["viajes"].classes_),
                format_func=FORMATOS["viajes"].get,
            )
            datos["horas_extra"] = st.selectbox(
                ETIQUETAS["horas_extra"],
                list(codificadores["horas_extra"].classes_),
                format_func=FORMATOS["horas_extra"].get,
            )
            datos["satisfaccion_entorno"] = st.selectbox(
                ETIQUETAS["satisfaccion_entorno"],
                list(codificadores["satisfaccion_entorno"].classes_),
                format_func=FORMATOS["satisfaccion_entorno"].get,
            )

        with c2:
            datos["satisfaccion_trabajo"] = st.selectbox(
                ETIQUETAS["satisfaccion_trabajo"],
                list(codificadores["satisfaccion_trabajo"].classes_),
                format_func=FORMATOS["satisfaccion_trabajo"].get,
            )
            datos["satisfaccion_companeros"] = st.selectbox(
                ETIQUETAS["satisfaccion_companeros"],
                list(codificadores["satisfaccion_companeros"].classes_),
                format_func=FORMATOS["satisfaccion_companeros"].get,
            )
            datos["implicacion"] = st.selectbox(
                ETIQUETAS["implicacion"],
                list(codificadores["implicacion"].classes_),
                format_func=FORMATOS["implicacion"].get,
            )

        with c3:
            datos["evaluacion"] = st.selectbox(
                ETIQUETAS["evaluacion"],
                list(codificadores["evaluacion"].classes_),
                format_func=FORMATOS["evaluacion"].get,
            )

        enviado = st.form_submit_button("🔮 Predecir abandono", type="primary")

    # --- Resultado -----------------------------------------------------------
    if enviado:
        try:
            resultado, prob_no, prob_yes = predecir(datos, artefactos)
        except Exception as exc:  # pragma: no cover - defensivo
            st.error(f"Ocurrió un error al predecir: {exc}")
            st.stop()

        st.divider()
        st.subheader("Resultado")

        col_resultado, col_prob = st.columns([1, 2])

        with col_resultado:
            if resultado == "Yes":
                st.error("🔴 **Abandonará** la empresa")
            else:
                st.success("🟢 **No abandonará** la empresa")

        with col_prob:
            c_no, c_yes = st.columns(2)
            c_no.metric("Probabilidad de permanencia", f"{prob_no:.1%}")
            c_yes.metric("Probabilidad de abandono", f"{prob_yes:.1%}")

        with st.expander("Ver datos ingresados"):
            st.dataframe(pd.DataFrame([datos])[columnas])


if __name__ == "__main__":
    main()
