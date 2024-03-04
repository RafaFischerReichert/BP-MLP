import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 0)

# Carregando dados e preenchendo valores vazios
data = pd.read_csv('car_prices.csv', encoding='utf-8')
data[['make', 'model', 'trim', 'state', 'color', 'interior', 'seller', 'transmission']].fillna('unknown', inplace=True)
data.dropna(inplace=True)

# Truncando os dados
truncated_data = data.sample(n=10000, random_state=42)

# Excluindo ids e datas e gerando one-hot encoding
truncated_data.drop(columns=['vin', 'saledate'], axis=1, inplace=True)
truncated_data = pd.get_dummies(truncated_data)

print(truncated_data.columns)

truncated_data.to_csv('processed_prices.csv')