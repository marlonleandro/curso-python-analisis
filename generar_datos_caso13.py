# -*- coding: utf-8 -*-
"""
Genera 3 archivos CSV de ejemplo para el Caso 13:
"Transformación e integración de datos con pandas".

Los datos incluyen problemas de calidad intencionales para practicar:
    - valores nulos
    - registros duplicados
    - inconsistencias de formato (mayúsculas/espacios, fechas, valores fuera de rango)

Archivos generados en ./data:
    ventas.csv      -> ~500 registros (transacciones de venta)
    clientes.csv    -> ~320 registros (dimensión de clientes)
    productos.csv   -> ~300 registros (dimensión de productos)

Ejecutar una sola vez:
    python generar_datos_caso13.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(20240901)  # semilla fija para reproducibilidad

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 1) CLIENTES  (dimensión)
# ----------------------------------------------------------------------
N_CLIENTES = 320

nombres_h = ["Juan", "Luis", "Carlos", "Pedro", "Jorge", "Miguel", "Diego",
             "Andrés", "Fernando", "Ricardo", "Raúl", "Alberto", "César",
             "Hugo", "Óscar", "Marco", "Sergio", "Walter", "Renzo", "Giancarlo"]
nombres_m = ["María", "Ana", "Rosa", "Lucía", "Carmen", "Sofía", "Valeria",
             "Camila", "Gabriela", "Adriana", "Daniela", "Milagros",
             "Fiorella", "Ximena", "Alejandra", "Estefanía", "Karla",
             "Patricia", "Rocío", "Mónica"]
apellidos = ["Quispe", "Flores", "García", "Rodríguez", "Torres", "Rojas",
             "Vargas", "Díaz", "Castro", "Ramos", "Castillo", "Salazar",
             "Gutiérrez", "Chávez", "Medina", "Espinoza", "Vásquez",
             "Paredes", "Aguilar", "Núñez"]

nombres = [f"{RNG.choice(nombres_h)} {RNG.choice(apellidos)}" for _ in range(N_CLIENTES // 2)] + \
          [f"{RNG.choice(nombres_m)} {RNG.choice(apellidos)}" for _ in range(N_CLIENTES - N_CLIENTES // 2)]
RNG.shuffle(nombres)
nombres = list(dict.fromkeys(nombres))  # quitar repetidos
while len(nombres) < N_CLIENTES:
    nombres.append(f"Cliente {RNG.integers(1000, 9999)}")
nombres = nombres[:N_CLIENTES]

ciudades = ["LIMA", "AREQUIPA", "TRUJILLO", "CUSCO", "PIURA", "CHICLAYO"]
segmentos = ["REGULAR", "VIP", "NUEVO"]

clientes = pd.DataFrame({
    "id_cliente": [f"C{i:04d}" for i in range(1, N_CLIENTES + 1)],
    "nombre": nombres,
    "ciudad": RNG.choice(ciudades, N_CLIENTES),
    "edad": RNG.integers(18, 89, N_CLIENTES),
    "segmento": RNG.choice(segmentos, N_CLIENTES, p=[0.6, 0.2, 0.2]),
    "fecha_registro": RNG.choice(pd.date_range("2023-01-01", "2024-12-31"), N_CLIENTES),
})

# --- inyectar problemas de calidad ---
idx_ciudad_nulo = RNG.choice(N_CLIENTES, 15, replace=False)
clientes.loc[idx_ciudad_nulo, "ciudad"] = np.nan

idx_edad_nulo = RNG.choice(N_CLIENTES, 10, replace=False)
clientes.loc[idx_edad_nulo, "edad"] = np.nan

idx_edad_rara = RNG.choice(N_CLIENTES, 5, replace=False)
clientes.loc[idx_edad_rara, "edad"] = RNG.integers(101, 120, 5)

idx_ciudad_mal = RNG.choice(N_CLIENTES, 20, replace=False)
for i in idx_ciudad_mal:
    if pd.notna(clientes.loc[i, "ciudad"]):
        clientes.loc[i, "ciudad"] = " " + str(clientes.loc[i, "ciudad"]).lower()

idx_seg_mal = RNG.choice(N_CLIENTES, 25, replace=False)
for i in idx_seg_mal:
    clientes.loc[i, "segmento"] = str(clientes.loc[i, "segmento"]).lower()

clientes["fecha_registro"] = clientes["fecha_registro"].dt.strftime("%Y-%m-%d")
clientes.to_csv(DATA_DIR / "clientes.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------
# 2) PRODUCTOS  (dimensión)
# ----------------------------------------------------------------------
N_PRODUCTOS = 300

categorias = ["LAPTOP", "MONITOR", "ACCESORIO", "AUDIO", "TABLET", "CELULAR"]
modelos = ["Pro", "Air", "Plus", "Max", "Lite", "Ultra", "Basic", "Neo"]
marcas = ["TechNova", "AndinaTech", "PeruPC", "MaxDigital"]

precio_por_cat = {
    "LAPTOP": (1800, 4500),
    "MONITOR": (400, 1200),
    "ACCESORIO": (20, 250),
    "AUDIO": (60, 900),
    "TABLET": (300, 1200),
    "CELULAR": (500, 3500),
}

productos = pd.DataFrame({
    "id_producto": [f"P{i:03d}" for i in range(1, N_PRODUCTOS + 1)],
    "nombre_producto": "",
    "categoria": RNG.choice(categorias, N_PRODUCTOS),
    "precio_lista": 0.0,
    "costo": 0.0,
})

for i in range(N_PRODUCTOS):
    cat = productos.loc[i, "categoria"]
    lo, hi = precio_por_cat[cat]
    precio = float(RNG.integers(lo, hi + 1))
    productos.loc[i, "nombre_producto"] = f"{RNG.choice(marcas)} {cat} {RNG.choice(modelos)} {i+1:03d}"
    productos.loc[i, "precio_lista"] = precio
    productos.loc[i, "costo"] = round(precio * RNG.uniform(0.55, 0.72), 2)

# --- inyectar problemas de calidad ---
idx_cat_nulo = RNG.choice(N_PRODUCTOS, 8, replace=False)
productos.loc[idx_cat_nulo, "categoria"] = np.nan

idx_precio_nulo = RNG.choice(N_PRODUCTOS, 6, replace=False)
productos.loc[idx_precio_nulo, "precio_lista"] = np.nan

idx_cat_mal = RNG.choice(N_PRODUCTOS, 15, replace=False)
for i in idx_cat_mal:
    if pd.notna(productos.loc[i, "categoria"]):
        productos.loc[i, "categoria"] = " " + str(productos.loc[i, "categoria"]).lower()

# duplicados: copiamos 2 filas completas
dup_productos = productos.sample(2, random_state=7)
productos = pd.concat([productos, dup_productos], ignore_index=True)

productos.to_csv(DATA_DIR / "productos.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------
# 3) VENTAS  (tabla de hechos)
# ----------------------------------------------------------------------
N_VENTAS = 500

sucursales = ["LIMA", "AREQUIPA", "TRUJILLO", "CUSCO"]
metodos = ["EFECTIVO", "TARJETA", "TRANSFERENCIA"]

fechas_ventas = [pd.Timestamp(d).strftime("%Y-%m-%d")
                 for d in RNG.choice(pd.date_range("2024-01-01", "2024-06-30"), N_VENTAS)]

ventas = pd.DataFrame({
    "id_venta": [f"V{i:04d}" for i in range(1, N_VENTAS + 1)],
    "fecha": fechas_ventas,
    "id_cliente": RNG.choice(clientes["id_cliente"].to_numpy(), N_VENTAS),
    "id_producto": RNG.choice(productos["id_producto"].to_numpy()[:N_PRODUCTOS], N_VENTAS),
    "sucursal": RNG.choice(sucursales, N_VENTAS),
    "cantidad": RNG.integers(1, 11, N_VENTAS),
    "precio_unitario": RNG.integers(50, 3000, N_VENTAS).astype(float),
    "metodo_pago": RNG.choice(metodos, N_VENTAS, p=[0.5, 0.35, 0.15]),
})

# --- inyectar problemas de calidad ---
# nulos
ventas.loc[RNG.choice(N_VENTAS, 10, replace=False), "id_cliente"] = np.nan
ventas.loc[RNG.choice(N_VENTAS, 8, replace=False), "id_producto"] = np.nan
ventas.loc[RNG.choice(N_VENTAS, 12, replace=False), "cantidad"] = np.nan
ventas.loc[RNG.choice(N_VENTAS, 10, replace=False), "precio_unitario"] = np.nan
ventas.loc[RNG.choice(N_VENTAS, 10, replace=False), "metodo_pago"] = np.nan

# claves foráneas inexistentes
idx_bad = RNG.choice(N_VENTAS, 6, replace=False)
ventas.loc[idx_bad, "id_cliente"] = "C9999"

# sucursal inconsistente
idx_suc = RNG.choice(N_VENTAS, 30, replace=False)
for i in idx_suc:
    if pd.notna(ventas.loc[i, "sucursal"]):
        ventas.loc[i, "sucursal"] = " " + str(ventas.loc[i, "sucursal"]).lower()

# método de pago inconsistente
idx_met = RNG.choice(N_VENTAS, 15, replace=False)
for i in idx_met:
    if pd.notna(ventas.loc[i, "metodo_pago"]):
        ventas.loc[i, "metodo_pago"] = str(ventas.loc[i, "metodo_pago"]).lower()

# cantidades fuera de rango
idx_neg = RNG.choice(N_VENTAS, 5, replace=False)
ventas.loc[idx_neg, "cantidad"] = -RNG.integers(1, 6, 5)
idx_big = RNG.choice(N_VENTAS, 5, replace=False)
ventas.loc[idx_big, "cantidad"] = RNG.integers(50, 99, 5)

# fechas en formato inconsistente (dd/mm/yyyy), con día >= 13 para que sea inequívoco
idx_fecha = RNG.choice(N_VENTAS, 20, replace=False)
for i in idx_fecha:
    d = pd.Timestamp(ventas.loc[i, "fecha"])
    dia = int(RNG.integers(13, 29))
    ventas.loc[i, "fecha"] = f"{dia:02d}/{d.month:02d}/{d.year}"

# fechas corruptas
idx_badfecha = RNG.choice(N_VENTAS, 3, replace=False)
ventas.loc[idx_badfecha, "fecha"] = "no registrada"

# duplicados: copiamos 12 filas completas
dup_ventas = ventas.sample(12, random_state=123)
ventas = pd.concat([ventas, dup_ventas], ignore_index=True)

ventas.to_csv(DATA_DIR / "ventas.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------
# Resumen de lo generado
# ----------------------------------------------------------------------
print("Archivos generados en ./data :")
print(f"  ventas.csv    -> {ventas.shape[0]} filas x {ventas.shape[1]} columnas")
print(f"  clientes.csv  -> {clientes.shape[0]} filas x {clientes.shape[1]} columnas")
print(f"  productos.csv -> {productos.shape[0]} filas x {productos.shape[1]} columnas")
print("\nNulos en ventas:\n", ventas.isna().sum())
print("\nNulos en clientes:\n", clientes.isna().sum())
print("\nNulos en productos:\n", productos.isna().sum())
