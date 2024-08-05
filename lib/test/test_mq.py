import os
import pandas as pd
import yfinance as yf
from lib.volatilidades.rolling_forecast import calculate_rolling_volatility
from lib.auxiliares.VaR import var_vol
from lib.auxiliares.esReal import es_real
from lib.auxiliares.ES import expected_shortfall
from lib.backtest.MQBacktest import *
from lib.auxiliares.simulateReturns import simulate_returns

nivel_confianza = 0.975
indexes = ['SAN.MC', 'BBVA.MC', 'SAB.MC', '^IBEX', 'BBVAE.MC', 'XTC5.MI', 'EURUSD=X']
first_date = '2021-06-28'
start_date = '2023-06-28'
end_date = '2024-06-28'
horizontes = [1, 10]

# Crear DataFrame inicial con las columnas necesarias
columnas = ['Real ES', 'Real Excepciones']
volatilidades_all = ['GJR_GARCH', 'EWMA', 'PERCEPTRON', 'LSTM', 'RANDOM_FOREST']
output_vol = './output/vol_forecasting_mq.xlsx'
os.makedirs(os.path.dirname(output_vol), exist_ok=True)

for vol in volatilidades_all:
    for horizonte in horizontes:
        columnas.extend([f'{vol} {horizonte}d ES', f'{vol} {horizonte}d Excepciones'])
df = pd.DataFrame(columns=columnas, index=indexes)

with pd.ExcelWriter(output_vol, engine='xlsxwriter') as writer:
    for index in indexes:
        es, exceptions = es_real(index, nivel_confianza, start_date, end_date)
        df.loc[index, 'Real ES'] = es
        df.loc[index, 'Real Excepciones'] = exceptions
        # Recuperamos los datos históricos
        data = yf.download(index, first_date, end_date)
        # Calcular los retornos logarítmicos
        data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        # Los almacenamos en un array limpiando los NA (debería ser sólo la primera fecha)
        returns = data['Log Returns'].dropna()
        retornos_reales = returns[start_date:end_date]
        # Calculamos la volatilidad para las fechas deseadas
        for horizonte in horizontes:
            volatilidades_all = calculate_rolling_volatility(returns, start_date, end_date, horizonte)

            # Esto sólo es para imprimir volatilidades
            volatilidades_excel = volatilidades_all.copy()
            for col in volatilidades_excel.select_dtypes(include=[np.number]).columns:
                volatilidades_excel[col] = volatilidades_excel[col].apply(lambda x: f"{x:.6f}".replace('.', ','))
                volatilidades_excel.to_excel(writer, sheet_name=f'{index} {horizonte} días')
                workbook = writer.book
                worksheet = writer.sheets[f'{index} {horizonte} días']

            for vol in volatilidades_all:
                volatilidades = volatilidades_all[vol]

                var = var_vol(volatilidades, nivel_confianza)
                es = expected_shortfall(volatilidades, nivel_confianza)

                es_test = EStestMultiQuantile(retornos_reales, lambda: simulate_returns(volatilidades, retornos_reales),
                                      1 - nivel_confianza,
                                      var, es, 1000, 1 - nivel_confianza)

                col_es = f'{vol} {horizonte}d ES'
                col_ex = f'{vol} {horizonte}d Excepciones'
                df.loc[index, col_es] = es[-1]
                df.loc[index, col_ex] = es_test.salida()
            print("Ejecutado horizonte: " + str(horizonte) + "d")
        print("Ejecutado índice completo: " + index)

# Convertir cualquier estructura compleja en tipo manejable (float)
df = df.applymap(lambda x: float(x) if isinstance(x, (np.ndarray, list)) else float(x))

# Convertir los datos a formato numérico con coma como separador decimal
output_path = './output/MQ_backtest_forecasting.xlsx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Guardar el DataFrame en Excel con formato numérico y coma como separador decimal
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Resultados')
    workbook = writer.book
    worksheet = writer.sheets['Resultados']
    num_format = workbook.add_format({'num_format': '#,##0.00'})
    percentage_format = workbook.add_format({'num_format': '0.00%'})
    integer_format = workbook.add_format({'num_format': '0'})

    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num + 1, value)

    for row_num, row_data in enumerate(df.values):
        for col_num, cell_data in enumerate(row_data):
            if 'ES' in df.columns[col_num]:  # Formatear las columnas ES como porcentaje
                worksheet.write(row_num + 1, col_num + 1, cell_data, percentage_format)
            elif 'Excepciones' in df.columns[col_num]:  # Formatear las columnas de excepciones como enteros
                worksheet.write(row_num + 1, col_num + 1, cell_data, integer_format)
            else:
                worksheet.write(row_num + 1, col_num + 1, cell_data, num_format)

print(f"Comparative results have been saved to {output_path}")
