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
15. Analisis y visualizacion avanzada de datos.
16. Introduccion al Machine Learning: clasificacion con k-NN, evaluacion y publicacion del modelo.
17. Caso practico: capacidad de red de AndesTel (trafico de red y analisis por zona).
18. Proyecto final: TelcoNova, analisis de churn, modelos predictivos y priorizacion de una campaña de retencion.
19. Caso de fraude financiero: BAF, riesgo de solicitudes de apertura de cuenta y priorizacion de revision.

## Recursos incluidos

- Notebooks de trabajo con ejemplos y ejercicios practicos.
- Archivos CSV de alumnos, ventas, encuestas y datos censales.
- Archivos CSV de casos practicos: abandono de empleados, ventas retail y transformacion/integracion de datos (ventas, clientes y productos).
- Archivos CSV del caso AndesTel: caso1_clientes.csv, caso1_zonas.csv y caso1_trafico_red.csv.
- Archivos del challenge: student-data-v2.csv, mentor-feedback.csv y sample_submission_v2.csv.
- Archivos exportados en formatos CSV y JSON.
- Aplicaciones Streamlit (carpeta `app/`): caso AndesTel (`app/caso1_app.py`) y prediccion de abandono de empleados (`app/abandono_app.py`).
- Modelo de Machine Learning entrenado (`exports/modelo_abandono.joblib`) y Web API (`api_abandono.py`).
- Script de mejora y ampliacion del dataset de abandono (`mejorar_datos_abandono.py`).

## Challenge: TSS Pandas Challenge #2

Challenge resuelto en el notebook `14.Challenge_Pandas_Analytics.ipynb`, organizado por The Software Society (TSS). Consiste en limpiar, combinar y analizar los datasets `student-data-v2.csv` y `mentor-feedback.csv` siguiendo el flujo: cargar, inspeccionar, limpiar, combinar y analizar.

Preguntas resueltas:

1. Q1 - Valores nulos (15 pts): conteo total de valores faltantes del dataset.
2. Q2 - Duplicados (15 pts): eliminar filas duplicadas y contar las restantes.
3. Q3 - Estandarizacion de texto (20 pts): normalizar la columna `department` (espacios y mayusculas).
4. Q4 - Merge (15 pts): left join por `student_id` y conteo de feedback faltante.
5. Q5 - GroupBy (20 pts): promedio de `feedback_score` por departamento.
6. Q6 - Fechas (15 pts): convertir `join_date` a datetime y contar ingresos en agosto de 2023.

## Machine Learning (notebook 16)

El notebook `16.Introduccion_al_Machine_Learning.ipynb` introduce los conceptos basicos de Machine Learning (tipos de entrenamiento y flujo de trabajo) y resuelve un caso de **clasificacion con k-NN**: predecir si un empleado abandonara su trabajo a partir del dataset `abandono_empleados.csv`.

Flujo resuelto:

1. Cargar y preparar datos.
2. Analisis exploratorio (EDA).
3. Codificar variables categoricas y segmentar datos (entrenamiento/prueba).
4. Entrenar el modelo k-NN.
5. Evaluar el modelo y hacer una prediccion con un dato real.
6. Publicar el modelo en una Web API con FastAPI (`api_abandono.py`), usando el modelo guardado en `exports/modelo_abandono.joblib`.

El dataset `abandono_empleados.csv` fue mejorado con el script `mejorar_datos_abandono.py`: la columna `sexo` se convirtio a `F`/`M` (femenino/masculino) y se añadieron 2000 registros sinteticos coherentes con los datos originales (3470 filas en total).

Ademas, en la carpeta `app/` hay una **aplicacion Streamlit** (`app/abandono_app.py`) que permite evaluar a un empleado desde un formulario: captura los datos con selectores y cajas de texto y muestra la prediccion de abandono junto con sus probabilidades, usando el modelo guardado en `exports/modelo_abandono.joblib`.

```bash
streamlit run app/abandono_app.py
```

## Caso practico: capacidad de red de AndesTel (notebook 17)

El notebook `17.CASO1_Trafico_red_telecom.ipynb` resuelve el caso de capacidad de red: determina en que zonas existe capacidad para vender nuevos servicios y en que zonas es necesario ampliar la infraestructura, usando el percentil 95 del trafico horario.

Ademas, en la carpeta `app/` hay una **aplicacion Streamlit** (`app/caso1_app.py`) que resuelve el mismo caso de forma interactiva: permite cargar los tres archivos (zonas, clientes y trafico), limpia y normaliza los datos, analiza por zona, genera graficos y muestra la recomendacion final.

```bash
streamlit run app/caso1_app.py
```

## Proyecto final: TelcoNova (notebook 18)

El notebook `18.Proyecto_Final_Caso_TelcoNova.ipynb` sigue primero la metodologia empresarial y exploratoria del notebook 17 y luego el flujo de Machine Learning del notebook 16. Incluye limpieza, evolucion mensual, agrupaciones, visualizaciones, k-NN, regresion logistica, evaluacion en prueba reservada y priorizacion del 10 % de clientes con mayor riesgo.

Utiliza los archivos de [Telecom Churn Case Study Hackathon C33](https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-C33/data), ubicados en `data/telecom_churn/`: `train.csv`, `test.csv` y `data_dictionary.csv`. El archivo `sample.csv` es opcional. La descarga requiere una cuenta Kaggle y aceptar las reglas de la competencia.

Ejecutar desde la raiz del proyecto con Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn y Joblib. Los resultados se guardan en `exports/telconova/`: comparacion de modelos, evaluacion local, escenarios de capacidad, modelo serializado, resumen reproducible y, si existe `test.csv`, ranking y lista de contactos. El notebook explica el supuesto temporal necesario para interpretar el modelo como anticipacion y distingue riesgo de abandono de efecto de una campaña.

## Caso de fraude financiero: NovaFin (notebook 19)

El notebook `19.Caso_Fraude_Financiero.ipynb` sigue la metodologia de los notebooks 18, 17 y 16, en ese orden. Desarrolla un caso de riesgo de fraude en solicitudes de apertura de cuentas con `Base.csv` de [Bank Account Fraud (BAF)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), ubicado en `data/bank_account_fraud/`.

Incluye limpieza de marcadores de ausencia, EDA, separacion temporal (meses 0–4 para entrenar, 5 para validar y 6–7 para probar), comparacion de Dummy, k-NN, regresion logistica y arbol de decision. Por defecto ajusta los modelos con una muestra de 100 000 solicitudes del bloque de entrenamiento; mantiene completos los meses de validacion y prueba. Evalua la captura de fraude en el 5 % priorizado por mes y los errores por grupos.

Ejecutar desde la raiz con Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn y Joblib. Guarda modelo, funcion de preparacion, metricas y cola didactica en `exports/fraude_financiero/`. BAF contiene datos sinteticos para experimentacion: la salida estima riesgo de una solicitud y no determina intenciones individuales ni una fecha futura de fraude.

## Autor

Marlon Leandro
