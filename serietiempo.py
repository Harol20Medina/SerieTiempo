import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")  # Ignorar advertencias para simplificar la salida

# **Definición de las Variables**
# 1. Variable Dependiente:
#    - `Casos Confirmados`: Valores acumulados por fecha que queremos modelar y predecir.
#    - Esto se toma de las columnas que contienen las fechas en los datos.
#
# 2. Variable Independiente:
#    - `Tiempo` (Fechas): Es implícita en el modelo de series de tiempo, ya que los valores dependen del orden temporal.
#
# **Por qué no se consideran las demás columnas:**
# - `Province/State` y `Country/Region`: Se usan solo para filtrar los datos (por ejemplo, por país o provincia).
# - `Lat` y `Long`: Aunque son variables geográficas útiles en análisis espaciales, no aportan valor directo a un modelo ARIMA,
#   que se enfoca en patrones temporales dentro de una única serie de datos.

# Cargar los datos desde un archivo CSV
# Justificación: Los datos son esenciales para crear una serie de tiempo que modela casos confirmados de COVID-19.
ruta_archivo = r'C:\Users\ben19\Downloads\codigoIA\time_series_covid_19_confirmed.csv'
datos_covid = pd.read_csv(ruta_archivo, delimiter=';')

# Mostrar los países disponibles en los datos
# Propósito: Permitir al usuario seleccionar un país para realizar el análisis.
print("Países disponibles en los datos:")
print(datos_covid['Country/Region'].unique())

# Seleccionar dinámicamente el país para el análisis
pais = input("Ingrese el nombre del país que desea analizar: ").strip()

# Verificar si el país ingresado existe en los datos
# Justificación: Garantizar que el país seleccionado esté disponible para evitar errores.
if pais not in datos_covid['Country/Region'].values:
    raise ValueError(f"El país '{pais}' no se encuentra en los datos.")

# Filtrar los datos para el país seleccionado
# Justificación: Los datos deben ser específicos del país para evitar mezclar información irrelevante.
datos_pais = datos_covid[datos_covid['Country/Region'] == pais]

# Transponer las columnas de fechas para crear una serie de tiempo
# Justificación: Transformamos los datos para que cada fecha sea un índice, lo que facilita el análisis temporal.
serie_tiempo = datos_pais.iloc[:, 4:-1].T  # Los datos de casos confirmados
serie_tiempo.index = pd.to_datetime(serie_tiempo.index, format='%m/%d/%y', errors='coerce')  # Convertir fechas
serie_tiempo.columns = ['Casos Confirmados']  # Renombrar la columna

# Eliminar fechas duplicadas en el índice para evitar errores
# Justificación: Un índice duplicado puede causar problemas al modelar o graficar.
serie_tiempo = serie_tiempo[~serie_tiempo.index.duplicated(keep='first')]

# Ajustar la frecuencia a diaria y rellenar valores faltantes con el último valor disponible
# Justificación: Asegura que la serie sea continua, algo fundamental para los modelos de series de tiempo.
serie_tiempo = serie_tiempo.asfreq('D').ffill()

# Graficar la evolución de casos confirmados en el tiempo
# Propósito: Identificar visualmente patrones como tendencias o cambios abruptos en los casos confirmados.
plt.figure(figsize=(12, 6))
plt.plot(serie_tiempo.index, serie_tiempo['Casos Confirmados'], label='Casos Confirmados', color='blue')
plt.title(f'Evolución de Casos Confirmados de COVID-19 en {pais}')
plt.xlabel('Fecha')
plt.ylabel('Casos Confirmados')
plt.grid(True)
plt.legend()
plt.show()

# Gráfico de Autocorrelación (ACF)
# Propósito: Identificar correlaciones en los rezagos de los datos para sugerir el parámetro q (orden del promedio móvil).
# Justificación: Ayuda a determinar cuántos valores previos influyen significativamente en los valores futuros.
plot_acf(serie_tiempo['Casos Confirmados'], lags=40)
plt.title("Gráfico de Autocorrelación (ACF)")
plt.xlabel("Rezagos")
plt.ylabel("Correlación")
plt.grid(True)
plt.show()

# Gráfico de Autocorrelación Parcial (PACF)
# Propósito: Identificar correlaciones directas entre un valor y sus rezagos para sugerir el parámetro p (términos autorregresivos).
# Justificación: Muestra qué rezagos tienen un impacto significativo sin considerar correlaciones intermedias.
plot_pacf(serie_tiempo['Casos Confirmados'], lags=40)
plt.title("Gráfico de Autocorrelación Parcial (PACF)")
plt.xlabel("Rezagos")
plt.ylabel("Correlación Parcial")
plt.grid(True)
plt.show()

# Dividir los datos en entrenamiento (80%) y prueba (20%)
# Propósito: Validar el modelo en datos que no se usaron para su ajuste.
tamano_entrenamiento = int(len(serie_tiempo) * 0.8)
entrenamiento, prueba = serie_tiempo.iloc[:tamano_entrenamiento], serie_tiempo.iloc[tamano_entrenamiento:]

# Selección automática del mejor modelo ARIMA basado en el AIC
# Propósito: Encontrar el modelo con mejor ajuste minimizando el AIC.
def seleccionar_modelo_arima(serie):
    mejor_aic = float('inf')  # Inicia con un valor muy alto
    mejor_orden = None
    mejor_modelo = None
    
    # Probar diferentes combinaciones de parámetros (p, d, q)
    for p in range(0, 5):  # Número de rezagos autorregresivos
        for d in range(0, 2):  # Grado de diferenciación
            for q in range(0, 5):  # Orden del promedio móvil
                try:
                    modelo = ARIMA(serie, order=(p, d, q)).fit()  # Ajustar modelo ARIMA
                    aic = modelo.aic  # Criterio AIC
                    if aic < mejor_aic:  # Guardar el modelo con el menor AIC
                        mejor_aic = aic
                        mejor_orden = (p, d, q)
                        mejor_modelo = modelo
                except:
                    continue
    return mejor_orden, mejor_modelo

# Encontrar el mejor modelo y sus parámetros óptimos (p, d, q)
# Justificación: Un modelo optimizado mejora la precisión de las predicciones.
orden_optimo, modelo_optimo = seleccionar_modelo_arima(entrenamiento['Casos Confirmados'])
print(f"Mejor modelo ARIMA encontrado: {orden_optimo}")

# Realizar predicciones sobre el conjunto de prueba
# Justificación: Validamos el modelo comparando predicciones con valores reales.
prediccion = modelo_optimo.forecast(steps=len(prueba))
prueba = prueba.copy()  # Crear una copia para evitar advertencias
prueba['Casos Predichos'] = prediccion.values

# Calcular métricas de error para evaluar el modelo
# Propósito: Medir qué tan bien se ajusta el modelo a los datos de prueba.
mse = mean_squared_error(prueba['Casos Confirmados'], prueba['Casos Predichos'])  # Error Cuadrático Medio
mae = mean_absolute_error(prueba['Casos Confirmados'], prueba['Casos Predichos'])  # Error Absoluto Medio
print(f'Error Cuadrático Medio (MSE): {mse}')
print(f'Error Absoluto Medio (MAE): {mae}')

# Graficar las predicciones frente a los datos reales
# Propósito: Evaluar visualmente la precisión de las predicciones del modelo.
plt.figure(figsize=(12, 6))
plt.plot(entrenamiento.index, entrenamiento['Casos Confirmados'], label='Datos de Entrenamiento')
plt.plot(prueba.index, prueba['Casos Confirmados'], label='Datos de Prueba', color='green')
plt.plot(prueba.index, prueba['Casos Predichos'], label='Predicciones', color='red')
plt.title(f'Predicciones ARIMA para COVID-19 en {pais}')
plt.xlabel('Fecha')
plt.ylabel('Casos Confirmados')
plt.legend()
plt.grid(True)
plt.show()

# Graficar residuos para evaluar la calidad del modelo
# Propósito: Verificar si los residuos son ruido blanco (sin patrones ni tendencias).
residuos = modelo_optimo.resid
plt.figure(figsize=(12, 6))
plt.plot(residuos, label='Residuos')
plt.axhline(0, linestyle='--', color='red')  # Línea de referencia en cero
plt.title('Residuos del Modelo')
plt.xlabel('Fecha')
plt.ylabel('Residuos')
plt.grid(True)
plt.legend()
plt.show()

# Guardar los datos procesados y las predicciones
# Justificación: Permite análisis posteriores o compartir
