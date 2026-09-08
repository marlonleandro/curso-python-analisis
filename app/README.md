# Aplicación Streamlit — Caso 1: Capacidad de red de AndesTel

Aplicación interactiva que resuelve el **Caso 1** del notebook
`CASO1.Trafico_red_telecom.ipynb`:

1. Carga los tres archivos (zonas, clientes y tráfico de red). Los archivos
   pueden variar con el tiempo: la app detecta las columnas por nombre
   (ignora mayúsculas, acentos, espacios y guiones).
2. Limpia y normaliza los datos: nulos, duplicados, texto en mayúsculas,
   fechas y datos incoherentes.
3. Analiza por zona: clientes por zona, clientes por plan, tráfico por hora.
4. Genera los gráficos de tráfico y capacidad.
5. Muestra la **recomendación final** (punto 19 del notebook), clasificando
   cada zona como 🟢 Saludable, 🟡 Preventivo o 🔴 Crítico.

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la raíz del proyecto (`d:\Clases\NEWHORIZONS\Python\Ejercicios`):

```bash
streamlit run app/caso1_app.py
```

La aplicación se abre en el navegador con los **datos de ejemplo** ya cargados.
También puedes subir tus propios CSV desde la barra lateral.

## Política de capacidad

| Utilización P95 | Estado | Decisión |
|---:|---|---|
| ≤ 65% | 🟢 Saludable | Se pueden captar nuevos clientes |
| >65% y ≤80% | 🟡 Preventivo | Venta controlada |
| >80% | 🔴 Crítico | No vender hasta ampliar capacidad |
