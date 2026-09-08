"""
Mejora del dataset `data/abandono_empleados.csv`
=================================================

1. Convierte la columna `sexo` a las categorías F (femenino) y M (masculino)
   según el mapeo: 0=F, 1=M, 2=F, 3=M, 4=F.
2. Añade 2000 registros sintéticos coherentes con los datos existentes
   (muestreo bootstrap por filas + ruido gaussiano en las variables numéricas,
   recortado a los rangos originales).

Ejecutar desde la raíz del proyecto:

    python mejorar_datos_abandono.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RUTA = Path("data/abandono_empleados.csv")
SEMILLA = 42
N_NUEVOS = 2000

# Mapeo de códigos de sexo -> categoría
MAPA_SEXO = {0.0: "F", 1.0: "M", 2.0: "F", 3.0: "M", 4.0: "F"}

# Variables numéricas a las que se añadirá ruido (se excluyen id y constantes).
NUMERICAS = [
    "edad",
    "distancia_casa",
    "nivel_laboral",
    "salario_mes",
    "num_empresas_anteriores",
    "incremento_salario_porc",
    "nivel_acciones",
    "anos_experiencia",
    "num_formaciones_ult_ano",
    "anos_compania",
    "anos_desde_ult_promocion",
    "anos_con_manager_actual",
]


def main() -> None:
    rng = np.random.default_rng(SEMILLA)
    data = pd.read_csv(RUTA, sep=";")

    # ------------------------------------------------------------------
    # 1. Transformar la columna sexo
    # ------------------------------------------------------------------
    data["sexo"] = data["sexo"].map(MAPA_SEXO)

    # Imputar los nulos respetando la proporción F/M observada.
    nulos_sexo = data["sexo"].isna()
    if nulos_sexo.any():
        valores = data.loc[~nulos_sexo, "sexo"].to_numpy()
        data.loc[nulos_sexo, "sexo"] = rng.choice(
            valores, size=int(nulos_sexo.sum())
        )

    # ------------------------------------------------------------------
    # 2. Generar los 2000 registros sintéticos
    # ------------------------------------------------------------------
    # Referencia "limpia": filas sin "#N/D" en las columnas categóricas clave.
    mascara_limpia = (
        (data["educacion"] != "#N/D")
        & (data["implicacion"] != "#N/D")
        & (data["satisfaccion_trabajo"] != "#N/D")
    )
    ref = data[mascara_limpia].reset_index(drop=True)

    # Distribuciones válidas para las columnas con valores especiales.
    conciliacion_valida = data.loc[data["conciliacion"] != "#N/D", "conciliacion"]
    anos_en_puesto_valida = data["anos_en_puesto"].dropna()

    # Rangos y desviación estándar de cada variable numérica.
    limites = {c: (data[c].min(), data[c].max()) for c in NUMERICAS}
    desviaciones = {c: ref[c].std() for c in NUMERICAS}

    filas_nuevas = []
    for n in range(N_NUEVOS):
        fila = ref.iloc[int(rng.integers(0, len(ref)))].copy()

        # Ruido gaussiano recortado a los rangos originales.
        for c in NUMERICAS:
            sigma = desviaciones[c] * 0.2
            nuevo = float(fila[c]) + float(rng.normal(0.0, sigma))
            lo, hi = limites[c]
            fila[c] = int(round(min(max(nuevo, lo), hi)))

        # conciliacion: si la fila muestreada traía "#N/D", usar un valor válido.
        if fila["conciliacion"] == "#N/D":
            fila["conciliacion"] = str(rng.choice(conciliacion_valida.to_numpy()))

        # anos_en_puesto: no puede superar los años en la empresa.
        if pd.isna(fila["anos_en_puesto"]):
            fila["anos_en_puesto"] = float(rng.choice(anos_en_puesto_valida.to_numpy()))
        fila["anos_en_puesto"] = float(min(float(fila["anos_en_puesto"]), int(fila["anos_compania"])))

        # Identificador nuevo y columnas constantes.
        fila["id"] = int(data["id"].max()) + n + 1
        fila["mayor_edad"] = "Y"
        fila["empleados"] = 1
        fila["horas_quincena"] = 80

        filas_nuevas.append(fila)

    nuevos = pd.DataFrame(filas_nuevas, columns=data.columns)

    # ------------------------------------------------------------------
    # 3. Combinar y guardar
    # ------------------------------------------------------------------
    resultado = pd.concat([data, nuevos], ignore_index=True)

    # Verificaciones rápidas.
    assert resultado["id"].is_unique, "Hay identificadores duplicados"
    assert resultado["sexo"].isin(["F", "M"]).all(), "Quedan valores no válidos en sexo"

    resultado.to_csv(RUTA, sep=";", index=False)

    print(f"Registros originales : {len(data)}")
    print(f"Registros añadidos   : {len(nuevos)}")
    print(f"Total                : {len(resultado)}")
    print()
    print("sexo      :", dict(resultado["sexo"].value_counts()))
    print("abandono  :", dict(resultado["abandono"].value_counts()))
    print("nulos sexo:", int(resultado["sexo"].isna().sum()))
    print("ids nuevos :", int(nuevos["id"].min()), "-", int(nuevos["id"].max()))


if __name__ == "__main__":
    main()
