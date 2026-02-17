import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

usd = pd.read_csv("usd.csv", usecols=["Date","USD_TRY_Buying"])
inf = pd.read_excel("inflation.xlsx", usecols=["Tarih","tüfe"])

usd['date'] = pd.to_datetime(usd['Date'], errors='coerce')
inf['date'] = pd.to_datetime(inf['Tarih'], errors='coerce')

usd.dropna(subset=['date'], inplace=True)
inf.dropna(subset=['date'], inplace=True)

usd.rename(columns={'USD_TRY_Buying':'usd'}, inplace=True)
inf.rename(columns={'tüfe':'tufe'}, inplace=True)

start_date = max(usd['date'].min(), inf['date'].min())
end_date = min(usd['date'].max(), inf['date'].max())

usd_filtered = usd[(usd['date'] >= start_date) & (usd['date'] <= end_date)]
inf_filtered = inf[(inf['date'] >= start_date) & (inf['date'] <= end_date)]

df = pd.merge(usd_filtered, inf_filtered, on='date', how='inner')

X = sm.add_constant(df['tufe'])  # sabit terim ekle
y = df['usd']
model = sm.OLS(y, X).fit()
print(model.summary())

plt.figure(figsize=(8,5))
plt.scatter(df['tufe'], df['usd'], color='blue', label='Data')
coef = np.polyfit(df['tufe'], df['usd'], 1)
poly1d_fn = np.poly1d(coef)
plt.plot(df['tufe'], poly1d_fn(df['tufe']), color='red', label='Regression line')
plt.title("USD vs TÜFE")
plt.xlabel("TÜFE")
plt.ylabel("USD (TL)")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['date'], df['usd'], label='USD (TL)', color='blue')
plt.plot(df['date'], df['tufe'], label='TÜFE', color='green')
plt.title("USD ve TÜFE Zaman Serisi")
plt.xlabel("Tarih")
plt.ylabel("Değer")
plt.legend()

plt.show()

print("Columns in df:", df.columns)
print(df.head())

plt.figure(figsize=(8,5))
plt.scatter(df['tufe'], df['usd'], color='blue', alpha=0.7, edgecolors='k')
plt.xlabel('Inflation (TÜFE)')
plt.ylabel('USD/TRY')
plt.title('USD vs Inflation')
plt.grid(True)

plt.savefig('example_plot.png', dpi=150)
plt.show()
