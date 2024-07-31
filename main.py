import os
import yfinance as yf
from lib.volatilidades.rolling import calculate_rolling_volatility

from lib.auxiliares.VaR import var_vol
from lib.auxiliares.esReal import es_real
from lib.auxiliares.ES import expected_shortfall
from lib.backtest.MQBacktest import *
from lib.auxiliares.simulateReturns import simulate_returns

nivel_confianza = 0.975
index = 'SAN.MC'
first_date = '2021-05-28'
start_date = '2023-05-28'
end_date = '2024-06-28'
horizonte = 1
es_real(index, nivel_confianza, start_date, end_date)
# Recuperamos los datos históricos
data = yf.download(index, first_date, end_date)
# Calcular los retornos logarítmicos
data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
# Los almacenamos en un array limpiando los NA (debería ser sólo la primera fecha)
returns = data['Log Returns'].dropna()
retornos_pasados = returns[first_date:start_date]
sample_mu = np.mean(retornos_pasados)
sample_sigma = np.std(retornos_pasados)
retornos_reales = returns[start_date:end_date]
estimaciones = len(retornos_reales)
# Calculamos la volatilidad para las fechas deseadas
volatilidades_all = calculate_rolling_volatility(returns, start_date, end_date, horizonte)
volatilidades = volatilidades_all['EWMA']

var = var_vol(volatilidades, nivel_confianza)
var_array = np.array(var)
es = expected_shortfall(volatilidades, nivel_confianza)
es_array = np.array(es)

output_path = './output/comparativa_volatilidades.xlsx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save DataFrame to Excel with comma as decimal separator
volatilidades_all.to_excel(output_path, float_format="%.6f")

# Convert numbers to strings and replace '.' with ','
for col in volatilidades_all.select_dtypes(include=[np.number]).columns:
    volatilidades_all[col] = volatilidades_all[col].apply(lambda x: f"{x:.6f}".replace('.', ','))

# Save the DataFrame with the modified decimal separator to Excel
volatilidades_all.to_excel(output_path)

print(f"Comparative results have been saved to {output_path}")

print(es[-1:])

es_test = EStestMultiQuantile(retornos_reales, lambda: simulate_returns(volatilidades, retornos_reales),
                              1 - nivel_confianza, var_array, es_array, 1000, 1 - nivel_confianza)
es_test.print()
