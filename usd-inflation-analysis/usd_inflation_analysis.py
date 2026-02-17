import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import plotly.express as px

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

X = sm.add_constant(df['tufe'])
y = df['usd']
model = sm.OLS(y, X).fit()
print(model.summary())

plt.figure(figsize=(8,5))
plt.scatter(df['tufe'], df['usd'], color='blue', label='Data')
coef = np.polyfit(df['tufe'], df['usd'], 1)
poly1d_fn = np.poly1d(coef)
plt.plot(df['tufe'], poly1d_fn(df['tufe']), color='red', label='Regression line')
plt.title("USD vs Inflation")
plt.xlabel("Inflation")
plt.ylabel("USD (TL)")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['date'], df['usd'], label='USD (TL)', color='blue')
plt.plot(df['date'], df['tufe'], label='TÜFE', color='green')
plt.title("USD ve Inflation time series")
plt.xlabel("DATE")
plt.ylabel("VALUE")
plt.legend()
plt.show()

fig = px.scatter(
    df,
    x="tufe",
    y="usd",
    title="USD vs Inflation (Interactive)",
    labels={"tufe": "Inflation (TÜFE)", "usd": "USD/TRY"},
    trendline="ols"
)

fig.show()


