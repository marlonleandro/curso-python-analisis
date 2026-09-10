import numpy as np
import pandas as pd

def preparar_variables(tabla, reglas):
    faltantes = set(reglas['entrada']) - set(tabla.columns)
    if faltantes:
        raise ValueError(f'Faltan predictores: {sorted(faltantes)}')
    x = tabla[reglas['entrada']].copy()
    for columna in reglas['numericas']:
        convertido = pd.to_numeric(x[columna], errors='coerce')
        if (x[columna].notna() & convertido.isna()).any():
            raise ValueError(f'Texto no numérico en {columna}; revisar el archivo.')
        x[columna] = convertido.astype(float).replace([np.inf, -np.inf], np.nan)
    for columna in reglas['menos_uno_es_nulo']:
        x[columna] = x[columna].mask(x[columna] == -1)
    for columna in reglas['negativo_es_nulo']:
        x[columna] = x[columna].mask(x[columna] < 0)
    for columna in reglas['categoricas']:
        normalizada = x[columna].astype('string').str.strip().str.upper()
        normalizada = normalizada.mask(normalizada == '')
        x[columna] = normalizada.astype(object).where(normalizada.notna(), np.nan)
    return x
