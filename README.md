# Curso de Python para Analisis de Datos

Repositorio de material de trabajo del curso de Python orientado al analisis de datos. El curso avanza desde los fundamentos del lenguaje hasta la manipulacion, agrupamiento y visualizacion de datos con las bibliotecas mas utilizadas.

## Contenido del curso

1. Introduccion a Python: variables, tipos de datos y estructuras de control.
2. Estructuras de datos: listas, tuplas, conjuntos y diccionarios.
3. Funciones: definicion, parametros, retorno de valores y alcance.
4. Funciones avanzadas y manejo de errores.
5. Modulos y librerias para analisis de datos.
6. Introduccion a NumPy y operaciones con arreglos.
7. Introduccion a Pandas y uso de DataFrames.
8. Lectura, manipulacion y exportacion de archivos con Pandas.
9. Agrupamiento y resumen de datos con Pandas.
10. Visualizacion de datos con Pandas, Matplotlib y Seaborn.
11. Caso practico de EDA: analisis de abandono de empleados.
12. Caso practico de EDA: analisis de ventas de una cadena retail.
13. Caso practico: limpieza, integracion y transformacion de datos con Pandas.
14. Challenge: limpieza, combinacion y analisis de datos con Pandas (TSS Pandas Challenge #2).

## Recursos incluidos

- Notebooks de trabajo con ejemplos y ejercicios practicos.
- Archivos CSV de alumnos, ventas, encuestas y datos censales.
- Archivos CSV de casos practicos: abandono de empleados, ventas retail y transformacion/integracion de datos (ventas, clientes y productos).
- Archivos del challenge: student-data-v2.csv, mentor-feedback.csv y sample_submission_v2.csv.
- Archivos exportados en formatos CSV y JSON.

## Challenge: TSS Pandas Challenge #2

Challenge resuelto en el notebook `14.Challenge_Pandas_Analytics.ipynb`, organizado por The Software Society (TSS). Consiste en limpiar, combinar y analizar los datasets `student-data-v2.csv` y `mentor-feedback.csv` siguiendo el flujo: cargar, inspeccionar, limpiar, combinar y analizar.

Preguntas resueltas:

1. Q1 - Valores nulos (15 pts): conteo total de valores faltantes del dataset.
2. Q2 - Duplicados (15 pts): eliminar filas duplicadas y contar las restantes.
3. Q3 - Estandarizacion de texto (20 pts): normalizar la columna `department` (espacios y mayusculas).
4. Q4 - Merge (15 pts): left join por `student_id` y conteo de feedback faltante.
5. Q5 - GroupBy (20 pts): promedio de `feedback_score` por departamento.
6. Q6 - Fechas (15 pts): convertir `join_date` a datetime y contar ingresos en agosto de 2023.

## Autor

Marlon Leandro
