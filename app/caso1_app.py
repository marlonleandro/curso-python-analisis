"""
Aplicación Streamlit — Caso 1: Capacidad de red de AndesTel
============================================================

Resuelve, paso a paso, el caso de capacidad de la red de AndesTel:

1. Carga los tres archivos (zonas, clientes y tráfico de red).
2. Limpia y normaliza los datos (nulos, duplicados, texto, fechas,
   datos incoherentes).
3. Analiza la información por zona.
4. Genera los gráficos de tráfico y capacidad.
5. Muestra la recomendación final (punto 19 del notebook).

Los archivos pueden variar con el tiempo: la aplicación detecta las
columnas por nombre (insensible a mayúsculas, espacios y guiones), por
lo que sigue funcionando aunque cambien ligeramente los encabezados.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Utilidades de mapeo de columnas
# ---------------------------------------------------------------------------
def normalizar_nombre_col(nombre: str) -> str:
    """Convierte un nombre de columna a minúsculas, sin acentos y con guiones bajos."""
    n = str(nombre)
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")
    n = n.strip().lower()
    n = re.sub(r"[^a-z0-9]+", "_", n)
    return n.strip("_")


# Nombres canónicos -> lista de posibles nombres que puede traer el archivo.
CAMPOS_ZONAS = {
    "zona_id": ["zona_id", "id_zona", "zonaid", "codigo_zona"],
    "zona": ["zona", "nombre_zona", "nombre", "zona_nombre"],
    "region": ["region", "sector", "sector_geografico"],
    "tecnologia": ["tecnologia", "tipo_red", "tecnologia_red"],
    "capacidad_mbps": ["capacidad_mbps", "capacidad", "cap_mbps", "capacidad_total"],
    "limite_operacion": ["limite_operacion", "limite_operativo", "limite"],
}

CAMPOS_CLIENTES = {
    "cliente_id": ["cliente_id", "id_cliente", "clienteid"],
    "zona_id": ["zona_id", "id_zona", "zonaid"],
    "plan_mbps": ["plan_mbps", "plan", "velocidad_mbps", "plan_contratado"],
    "estado": ["estado", "estatus", "estado_cliente"],
    "fecha_alta": ["fecha_alta", "fecha", "alta", "fecha_ingreso"],
}

CAMPOS_TRAFICO = {
    "timestamp": ["timestamp", "fecha_hora", "datetime", "tiempo"],
    "zona_id": ["zona_id", "id_zona", "zonaid"],
    "trafico_down_mbps": [
        "trafico_down_mbps",
        "trafico_down",
        "download_mbps",
        "down_mbps",
        "trafico_bajada_mbps",
    ],
    "trafico_up_mbps": [
        "trafico_up_mbps",
        "trafico_up",
        "upload_mbps",
        "up_mbps",
        "trafico_subida_mbps",
    ],
    "latencia_ms": ["latencia_ms", "latencia", "latencia_promedio_ms"],
    "perdida_paquetes_pct": [
        "perdida_paquetes_pct",
        "perdida_paquetes",
        "packet_loss_pct",
        "perdida_pct",
    ],
}


def mapear_columnas(df: pd.DataFrame, requeridas: dict) -> tuple[dict, list]:
    """Devuelve el mapeo campo canónico -> columna real y la lista de faltantes."""
    disponibles = {normalizar_nombre_col(c): c for c in df.columns}
    mapeo: dict[str, str | None] = {}
    faltantes: list[str] = []
    for campo, candidatos in requeridas.items():
        elegida = next((disponibles[c] for c in candidatos if c in disponibles), None)
        mapeo[campo] = elegida
        if elegida is None:
            faltantes.append(campo)
    return mapeo, faltantes


def asegurar_numericas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Fuerza el tipo numérico de las columnas indicadas."""
    for col in columnas:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def asegurar_fechas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Fuerza el tipo fecha de las columnas indicadas."""
    for col in columnas:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalizar_texto(df: pd.DataFrame, excluir: set[str]) -> pd.DataFrame:
    """Pone en mayúsculas y limpia espacios de las columnas de texto."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col in excluir:
            continue
        df[col] = df[col].astype(str).str.upper().str.strip()
    return df


def recomendar_venta(pct: float) -> str:
    """Clasifica la capacidad utilizada según la política de AndesTel (punto 19)."""
    if pd.isna(pct):
        return "⚪ Sin información (revisar capacidad)"
    if pct <= 65:
        return "🟢 Saludable (Se pueden captar nuevos clientes)"
    if pct <= 80:
        return "🟡 Preventivo (Venta controlada)"
    return "🔴 Crítico (No vender hasta ampliar capacidad)"


def estado_corto(pct: float) -> str:
    if pd.isna(pct):
        return "Sin datos"
    if pct <= 65:
        return "Saludable"
    if pct <= 80:
        return "Preventivo"
    return "Crítico"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def cargar_ejemplo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los archivos de ejemplo desde la carpeta data/."""
    rutas = {
        "clientes": DATA_DIR / "caso1_clientes.csv",
        "zonas": DATA_DIR / "caso1_zonas.csv",
        "trafico": DATA_DIR / "caso1_trafico_red.csv",
    }
    faltan = [r.name for r in rutas.values() if not r.exists()]
    if faltan:
        raise FileNotFoundError(
            "No se encontraron los archivos de ejemplo: " + ", ".join(faltan)
        )
    dfc = pd.read_csv(rutas["clientes"])
    dfz = pd.read_csv(rutas["zonas"])
    dft = pd.read_csv(rutas["trafico"])
    return dfc, dfz, dft


# ---------------------------------------------------------------------------
# Procesamiento (limpieza + análisis)
# ---------------------------------------------------------------------------
def procesar(
    df_clientes: pd.DataFrame, df_zonas: pd.DataFrame, df_trafico: pd.DataFrame
) -> dict:
    """Ejecuta todo el flujo del caso y devuelve los resultados paso a paso."""
    # --- 1. Mapear y renombrar columnas -----------------------------------
    mapeo_c, falt_c = mapear_columnas(df_clientes, CAMPOS_CLIENTES)
    mapeo_z, falt_z = mapear_columnas(df_zonas, CAMPOS_ZONAS)
    mapeo_t, falt_t = mapear_columnas(df_trafico, CAMPOS_TRAFICO)

    faltantes: dict[str, list] = {
        "clientes": falt_c,
        "zonas": falt_z,
        "tráfico": falt_t,
    }
    if any(faltantes.values()):
        raise ValueError(
            "Faltan columnas obligatorias: "
            + "; ".join(f"{k}: {', '.join(v)}" for k, v in faltantes.items() if v)
        )

    df_clientes = df_clientes.rename(columns={v: k for k, v in mapeo_c.items()})
    df_zonas = df_zonas.rename(columns={v: k for k, v in mapeo_z.items()})
    df_trafico = df_trafico.rename(columns={v: k for k, v in mapeo_t.items()})

    # --- 2. Tamaño original ----------------------------------------------
    shapes_original = {
        "clientes": df_clientes.shape,
        "zonas": df_zonas.shape,
        "tráfico": df_trafico.shape,
    }

    # --- 3. Valores nulos -------------------------------------------------
    nulos_antes = {
        "clientes": df_clientes.isnull().sum(),
        "zonas": df_zonas.isnull().sum(),
        "tráfico": df_trafico.isnull().sum(),
    }

    df_clientes = df_clientes.dropna()
    df_zonas = df_zonas.dropna()
    df_trafico = df_trafico.dropna()
    shapes_sin_nulos = {
        "clientes": df_clientes.shape,
        "zonas": df_zonas.shape,
        "tráfico": df_trafico.shape,
    }

    # --- 4. Duplicados ----------------------------------------------------
    duplicados_antes = {
        "clientes": int(df_clientes.duplicated().sum()),
        "zonas": int(df_zonas.duplicated().sum()),
        "tráfico": int(df_trafico.duplicated().sum()),
    }
    df_clientes = df_clientes.drop_duplicates()
    df_zonas = df_zonas.drop_duplicates()
    df_trafico = df_trafico.drop_duplicates()
    shapes_sin_duplicados = {
        "clientes": df_clientes.shape,
        "zonas": df_zonas.shape,
        "tráfico": df_trafico.shape,
    }

    # --- 5. Normalizar texto (mayúsculas / espacios) ----------------------
    df_clientes = normalizar_texto(df_clientes, excluir={"fecha_alta"})
    df_zonas = normalizar_texto(df_zonas, excluir=set())
    df_trafico = normalizar_texto(df_trafico, excluir={"timestamp"})

    # --- 6. Convertir fechas ---------------------------------------------
    df_clientes = asegurar_fechas(df_clientes, ["fecha_alta"])
    df_trafico = asegurar_fechas(df_trafico, ["timestamp"])

    # --- 7. Datos incoherentes -------------------------------------------
    df_clientes = asegurar_numericas(df_clientes, ["plan_mbps"])
    df_zonas = asegurar_numericas(df_zonas, ["capacidad_mbps", "limite_operacion"])
    for col in ["trafico_down_mbps", "trafico_up_mbps", "latencia_ms", "perdida_paquetes_pct"]:
        df_trafico = asegurar_numericas(df_trafico, [col])

    antes = {
        "clientes": df_clientes.shape[0],
        "zonas": df_zonas.shape[0],
        "tráfico": df_trafico.shape[0],
    }

    df_clientes = df_clientes[df_clientes["plan_mbps"] > 0]
    df_zonas = df_zonas[df_zonas["capacidad_mbps"] > 0]
    df_zonas = df_zonas[
        (df_zonas["limite_operacion"] >= 0) & (df_zonas["limite_operacion"] <= 1)
    ]
    df_trafico = df_trafico[df_trafico["trafico_down_mbps"] >= 0]
    df_trafico = df_trafico[df_trafico["trafico_up_mbps"] >= 0]
    df_trafico = df_trafico[df_trafico["latencia_ms"] >= 0]

    despues = {
        "clientes": df_clientes.shape[0],
        "zonas": df_zonas.shape[0],
        "tráfico": df_trafico.shape[0],
    }
    incoherentes = {k: antes[k] - despues[k] for k in antes}

    # --- 8. Análisis: clientes por zona ----------------------------------
    clientes_por_zona = (
        df_clientes.groupby("zona_id").size().reset_index(name="cantidad_clientes")
    )
    clientes_por_zona = clientes_por_zona.merge(
        df_zonas[["zona_id", "zona"]], on="zona_id", how="left"
    )

    clientes_por_zona_y_plan = df_clientes.pivot_table(
        index="zona_id",
        columns="plan_mbps",
        values="cliente_id",
        aggfunc="count",
        fill_value=0,
    )

    # --- 9. Análisis: tráfico por hora -----------------------------------
    df_trafico["fecha"] = df_trafico["timestamp"].dt.date
    df_trafico["hora"] = df_trafico["timestamp"].dt.hour
    df_trafico["hora_str"] = df_trafico["timestamp"].dt.strftime("%H:00")

    trafico_promedio_por_hora = (
        df_trafico.groupby(["hora", "hora_str"])["trafico_down_mbps"]
        .mean()
        .reset_index(name="promedio_trafico_down_mbps")
        .sort_values("hora")
    )

    trafico_promedio_por_hora_y_zona = df_trafico.pivot_table(
        index="hora",
        columns="zona_id",
        values="trafico_down_mbps",
        aggfunc="mean",
        fill_value=0,
    ).sort_index()
    # Para graficar con etiquetas de hora legibles.
    hora_a_str = trafico_promedio_por_hora.set_index("hora")["hora_str"].to_dict()
    trafico_promedio_por_hora_y_zona.index = [
        hora_a_str.get(h, f"{h:02d}:00") for h in trafico_promedio_por_hora_y_zona.index
    ]

    # --- 10. Análisis: tráfico agregado por zona --------------------------
    trafico_agrupado_por_zona = (
        df_trafico.groupby("zona_id")
        .agg(
            promedio_trafico_down_mbps=("trafico_down_mbps", "mean"),
            maximo_trafico_down_mbps=("trafico_down_mbps", "max"),
            percentil_95_trafico_down_mbps=("trafico_down_mbps", lambda x: x.quantile(0.95)),
            promedio_latencia_ms=("latencia_ms", "mean"),
            promedio_perdida_paquetes=("perdida_paquetes_pct", "mean"),
        )
        .reset_index()
    )

    # --- 11. Capacidad utilizada por zona ---------------------------------
    df_capacidad = trafico_agrupado_por_zona.merge(
        df_zonas[
            ["zona_id", "zona", "region", "tecnologia", "capacidad_mbps", "limite_operacion"]
        ],
        on="zona_id",
        how="left",
    )
    df_capacidad["capacidad_utilizada_pct"] = (
        df_capacidad["percentil_95_trafico_down_mbps"] / df_capacidad["capacidad_mbps"]
    ) * 100
    df_capacidad["capacidad_utilizada_pct"] = df_capacidad["capacidad_utilizada_pct"].round(2)
    df_capacidad["estado"] = df_capacidad["capacidad_utilizada_pct"].apply(estado_corto)
    df_capacidad["recomendacion_venta"] = df_capacidad["capacidad_utilizada_pct"].apply(
        recomendar_venta
    )
    df_capacidad = df_capacidad.sort_values(
        "capacidad_utilizada_pct", ascending=False, na_position="last"
    )

    return {
        "shapes_original": shapes_original,
        "nulos_antes": nulos_antes,
        "shapes_sin_nulos": shapes_sin_nulos,
        "duplicados_antes": duplicados_antes,
        "shapes_sin_duplicados": shapes_sin_duplicados,
        "incoherentes": incoherentes,
        "shapes_finales": despues,
        "df_clientes": df_clientes,
        "df_zonas": df_zonas,
        "df_trafico": df_trafico,
        "clientes_por_zona": clientes_por_zona,
        "clientes_por_zona_y_plan": clientes_por_zona_y_plan,
        "trafico_promedio_por_hora": trafico_promedio_por_hora,
        "trafico_promedio_por_hora_y_zona": trafico_promedio_por_hora_y_zona,
        "trafico_agrupado_por_zona": trafico_agrupado_por_zona,
        "df_capacidad": df_capacidad,
    }


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="AndesTel — Caso 1: Capacidad de red",
        page_icon="📡",
        layout="wide",
    )

    st.title("📡 AndesTel — Capacidad de red por zona")
    st.markdown(
        """
        **¿En qué zonas existe capacidad suficiente para continuar vendiendo nuevos
        servicios y en qué zonas es necesario ampliar la infraestructura antes de
        captar más clientes?**

        Esta aplicación resuelve el caso paso a paso: carga los tres archivos, limpia
        y normaliza los datos, analiza por zona, genera los gráficos y presenta la
        recomendación final.
        """
    )

    # --- Carga de archivos en la barra lateral ---------------------------
    with st.sidebar:
        st.header("1. Datos de entrada")
        origen = st.radio(
            "Origen de los datos",
            ["Datos de ejemplo", "Subir archivos"],
            help="Los archivos pueden variar con el tiempo; al subirlos se detectan las columnas automáticamente.",
        )

        df_clientes = df_zonas = df_trafico = None
        if origen == "Datos de ejemplo":
            try:
                df_clientes, df_zonas, df_trafico = cargar_ejemplo()
                st.success("Archivos de ejemplo cargados desde `data/`.")
            except FileNotFoundError as exc:
                st.error(str(exc))
        else:
            archivo_clientes = st.file_uploader("Clientes (CSV)", type=["csv"])
            archivo_zonas = st.file_uploader("Zonas (CSV)", type=["csv"])
            archivo_trafico = st.file_uploader("Tráfico de red (CSV)", type=["csv"])
            if archivo_clientes and archivo_zonas and archivo_trafico:
                df_clientes = pd.read_csv(archivo_clientes)
                df_zonas = pd.read_csv(archivo_zonas)
                df_trafico = pd.read_csv(archivo_trafico)
                st.success("Los 3 archivos se cargaron correctamente.")

        st.markdown("---")
        st.markdown(
            """
            **Política de capacidad**

            | Utilización P95 | Estado | Decisión |
            |---:|---|---|
            | ≤ 65% | 🟢 Saludable | Captar nuevos clientes |
            | >65% y ≤80% | 🟡 Preventivo | Venta controlada |
            | >80% | 🔴 Crítico | No vender hasta ampliar |
            """
        )

    # --- Si no hay datos, detener ---------------------------------------
    if df_clientes is None or df_zonas is None or df_trafico is None:
        st.info("👈 Sube los tres archivos CSV o elige **Datos de ejemplo** para comenzar.")
        st.stop()

    # --- Procesar --------------------------------------------------------
    try:
        res = procesar(df_clientes, df_zonas, df_trafico)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # --- Paso 2: carga ----------------------------------------------------
    tab_carga, tab_limpieza, tab_clientes, tab_trafico, tab_capacidad, tab_recomendacion = st.tabs(
        [
            "1. Carga",
            "2. Limpieza",
            "3. Clientes por zona",
            "4. Tráfico",
            "5. Capacidad",
            "6. Recomendación",
        ]
    )

    # ---------------------------------------------------------------- Carga
    with tab_carga:
        st.subheader("Carga de datos")
        st.markdown("Los archivos se cargan como `DataFrame` de pandas.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Clientes", f"{res['shapes_original']['clientes'][0]:,} filas")
        col2.metric("Zonas", f"{res['shapes_original']['zonas'][0]:,} filas")
        col3.metric("Tráfico de red", f"{res['shapes_original']['tráfico'][0]:,} filas")

        st.markdown("**Vista previa de los datos cargados**")
        st.markdown("Clientes")
        st.dataframe(df_clientes.head(), width="stretch")
        st.markdown("Zonas")
        st.dataframe(df_zonas.head(), width="stretch")
        st.markdown("Tráfico de red")
        st.dataframe(df_trafico.head(), width="stretch")

    # -------------------------------------------------------------- Limpieza
    with tab_limpieza:
        st.subheader("Limpieza y normalización")

        # 2.1 Nulos
        st.markdown("#### 2.1 Valores nulos")
        col1, col2, col3 = st.columns(3)
        for titulo, datos, col in [
            ("Clientes", res["nulos_antes"]["clientes"], col1),
            ("Zonas", res["nulos_antes"]["zonas"], col2),
            ("Tráfico", res["nulos_antes"]["tráfico"], col3),
        ]:
            with col:
                st.markdown(f"**{titulo}**")
                st.dataframe(
                    datos.rename("nulos").sort_values(ascending=False).to_frame(),
                    width="stretch",
                )
        st.markdown("Se eliminan las filas con valores nulos (`dropna`).")
        st.write(
            "Clientes:", res["shapes_original"]["clientes"], "→", res["shapes_sin_nulos"]["clientes"]
        )
        st.write(
            "Zonas:", res["shapes_original"]["zonas"], "→", res["shapes_sin_nulos"]["zonas"]
        )
        st.write(
            "Tráfico:", res["shapes_original"]["tráfico"], "→", res["shapes_sin_nulos"]["tráfico"]
        )

        # 2.2 Duplicados
        st.markdown("#### 2.2 Datos duplicados")
        st.write(
            "Filas duplicadas: "
            f"Clientes = {res['duplicados_antes']['clientes']}, "
            f"Zonas = {res['duplicados_antes']['zonas']}, "
            f"Tráfico = {res['duplicados_antes']['tráfico']}"
        )
        st.markdown("Se eliminan los duplicados (`drop_duplicates`).")
        st.write(
            "Clientes:", res["shapes_sin_nulos"]["clientes"], "→", res["shapes_sin_duplicados"]["clientes"]
        )
        st.write(
            "Zonas:", res["shapes_sin_nulos"]["zonas"], "→", res["shapes_sin_duplicados"]["zonas"]
        )
        st.write(
            "Tráfico:", res["shapes_sin_nulos"]["tráfico"], "→", res["shapes_sin_duplicados"]["tráfico"]
        )

        # 2.3 Normalización de texto
        st.markdown("#### 2.3 Normalización de texto")
        st.markdown(
            "Las columnas categóricas se convierten a **mayúsculas** y se eliminan "
            "los **espacios** sobrantes (p. ej. ` Activo ` → `ACTIVO`)."
        )
        st.markdown("Clientes")
        st.dataframe(res["df_clientes"].head(), width="stretch")
        st.markdown("Zonas")
        st.dataframe(res["df_zonas"].head(), width="stretch")
        st.markdown("Tráfico de red")
        st.dataframe(res["df_trafico"].head(), width="stretch")

        # 2.4 Fechas
        st.markdown("#### 2.4 Conversión de fechas")
        st.markdown(
            "`fecha_alta` (clientes) y `timestamp` (tráfico) se convierten a tipo fecha."
        )

        # 2.5 Incoherentes
        st.markdown("#### 2.5 Eliminación de datos incoherentes")
        st.markdown(
            """
            - `plan_mbps > 0`
            - `capacidad_mbps > 0`
            - `0 ≤ limite_operacion ≤ 1`
            - `trafico_down_mbps ≥ 0`, `trafico_up_mbps ≥ 0`, `latencia_ms ≥ 0`
            """
        )
        st.write(
            "Filas eliminadas por incoherencias: "
            f"Clientes = {res['incoherentes']['clientes']}, "
            f"Zonas = {res['incoherentes']['zonas']}, "
            f"Tráfico = {res['incoherentes']['tráfico']}"
        )
        st.write(
            "Tamaños finales: "
            f"Clientes = {res['shapes_finales']['clientes']}, "
            f"Zonas = {res['shapes_finales']['zonas']}, "
            f"Tráfico = {res['shapes_finales']['tráfico']}"
        )

    # ------------------------------------------------------ Clientes por zona
    with tab_clientes:
        st.subheader("Clientes por zona")

        st.markdown("#### Cantidad de clientes por zona")
        st.dataframe(res["clientes_por_zona"], width="stretch")

        fig = px.bar(
            res["clientes_por_zona"],
            x="zona_id",
            y="cantidad_clientes",
            color="zona",
            labels={"zona_id": "Zona", "cantidad_clientes": "Clientes"},
            title="Cantidad de clientes por zona",
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Clientes por zona y plan contratado (pivot)")
        st.dataframe(res["clientes_por_zona_y_plan"], width="stretch")

    # ----------------------------------------------------------------- Tráfico
    with tab_trafico:
        st.subheader("Tráfico de red")

        st.markdown("#### Tráfico de bajada promedio por hora del día")
        st.dataframe(res["trafico_promedio_por_hora"], width="stretch")

        fig = px.line(
            res["trafico_promedio_por_hora"],
            x="hora_str",
            y="promedio_trafico_down_mbps",
            markers=True,
            labels={
                "hora_str": "Hora",
                "promedio_trafico_down_mbps": "Promedio tráfico down (Mbps)",
            },
            title="Tráfico de bajada promedio por hora del día",
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Tráfico de bajada promedio por hora y por zona")
        st.dataframe(res["trafico_promedio_por_hora_y_zona"], width="stretch")

        fig = px.line(
            res["trafico_promedio_por_hora_y_zona"],
            markers=True,
            labels={"value": "Promedio tráfico down (Mbps)", "index": "Hora"},
            title="Tráfico de bajada promedio por hora del día, por zona",
        )
        fig.update_layout(
            xaxis_title="Hora", yaxis_title="Promedio tráfico down (Mbps)", legend_title="Zona"
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Tráfico agregado por zona")
        st.dataframe(res["trafico_agrupado_por_zona"], width="stretch")

    # --------------------------------------------------------------- Capacidad
    with tab_capacidad:
        st.subheader("Capacidad utilizada por zona")

        df_cap = res["df_capacidad"].copy()
        st.markdown(
            "La capacidad utilizada se calcula como: "
            "**P95 del tráfico de bajada ÷ capacidad del enlace × 100**."
        )
        mostrar = df_cap[
            [
                "zona_id",
                "zona",
                "region",
                "tecnologia",
                "capacidad_mbps",
                "percentil_95_trafico_down_mbps",
                "capacidad_utilizada_pct",
                "estado",
            ]
        ]
        st.dataframe(mostrar, width="stretch")

        colores = {"Saludable": "#2ecc71", "Preventivo": "#f1c40f", "Crítico": "#e74c3c", "Sin datos": "#95a5a6"}
        fig = go.Figure()
        fig.add_bar(
            x=df_cap["zona_id"],
            y=df_cap["capacidad_utilizada_pct"],
            marker_color=[colores.get(e, "#95a5a6") for e in df_cap["estado"]],
            text=df_cap["capacidad_utilizada_pct"],
            textposition="outside",
            name="% utilizado",
        )
        fig.add_hline(y=65, line_dash="dash", line_color="#2ecc71", annotation_text="65% · saludable")
        fig.add_hline(y=80, line_dash="dash", line_color="#e74c3c", annotation_text="80% · crítico")
        fig.update_layout(
            title="Capacidad utilizada por zona (P95)",
            xaxis_title="Zona",
            yaxis_title="Capacidad utilizada (%)",
            yaxis_range=[0, max(df_cap["capacidad_utilizada_pct"].max() * 1.2, 100)],
        )
        st.plotly_chart(fig, width="stretch")

    # ----------------------------------------------------------- Recomendación
    with tab_recomendacion:
        st.subheader("Recomendación final (punto 19 del notebook)")

        df_rec = res["df_capacidad"].copy()
        columnas = [
            "zona_id",
            "zona",
            "region",
            "tecnologia",
            "capacidad_utilizada_pct",
            "recomendacion_venta",
        ]
        st.dataframe(df_rec[columnas], width="stretch")

        conteo = df_rec["estado"].value_counts().reindex(
            ["Saludable", "Preventivo", "Crítico", "Sin datos"], fill_value=0
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Saludable", int(conteo["Saludable"]))
        c2.metric("🟡 Preventivo", int(conteo["Preventivo"]))
        c3.metric("🔴 Crítico", int(conteo["Crítico"]))
        c4.metric("⚪ Sin datos", int(conteo["Sin datos"]))

        fig = px.pie(
            names=conteo.index,
            values=conteo.values,
            color=conteo.index,
            color_discrete_map=colores,
            title="Distribución del estado de las zonas",
            hole=0.4,
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("### Resumen ejecutivo")
        criticas = df_rec[df_rec["estado"] == "Crítico"]
        preventivas = df_rec[df_rec["estado"] == "Preventivo"]
        saludables = df_rec[df_rec["estado"] == "Saludable"]

        if not criticas.empty:
            st.error(
                "🔴 **Ampliar infraestructura antes de vender en:** "
                + ", ".join(criticas["zona"].fillna(criticas["zona_id"]).astype(str))
                + "."
            )
        if not preventivas.empty:
            st.warning(
                "🟡 **Venta controlada en:** "
                + ", ".join(preventivas["zona"].fillna(preventivas["zona_id"]).astype(str))
                + "."
            )
        if not saludables.empty:
            st.success(
                "🟢 **Se pueden captar nuevos clientes en:** "
                + ", ".join(saludables["zona"].fillna(saludables["zona_id"]).astype(str))
                + "."
            )


if __name__ == "__main__":
    main()
