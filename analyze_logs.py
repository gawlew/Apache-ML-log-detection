import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import IsolationForest
import sys
sys.stdout.reconfigure(encoding='utf-8')

#ściezka do pliku tekstowego z logami
log_file = "logs/logs.txt"
#tabela na dane
data = []
#wzór regular expression do oddzielenia danych z logu
log_pattern = re.compile(
    r'(\S+) (\S+) (\S+) \[(.*?)\] "(.*?)" (\d{3}) (\S+)'
)
#otwieram plik tekstowy dla każdej linijki sprawdzam czy pasuje do wzoru regex jeśli tak zapisuje w data
#jako listę osobnych wyrazów
with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        match = log_pattern.match(line)
        if match:
            data.append(match.groups())
#definuje kolumny dla dataframea i tworze go z naszych danych
columns = ["ip", "client_id", "user_id", "datetime", "request", "status", "size"]
df = pd.DataFrame(data, columns=columns)
#zamieniamy datetime z wyrazu na rzeczywistą datę
df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%b/%Y:%H:%M:%S %z", errors="coerce")
#sort_index() sortuje dane po indexie nie po danych lub jak w tym przypadku po etykietach
df["hour"] = df["datetime"].dt.hour
requests_per_hour = df["hour"].value_counts().sort_index()
#get_dummies zamienia wartości etykiet na one hot np. POST 0 1 0
df["request_type"] = df["request"].str.split().str[0]
# Wyodrębnienie typu żądania HTTP (GET, POST, PUT itd.)
df["method"] = df["request"].str.split().str[0]
df = pd.get_dummies(df, columns=["request_type"],prefix="req", drop_first=False)
#brakujące wartości w logach w kolumnie size sa oznaczane jako - co powoduje błąd
#ponieważ nie możemy zamienic tego na int dlatego zamieniamy - na np.nan
#i usuwamy wartosci które nie mają rozmiaru
df["size"] = df["size"].replace("-", np.nan).astype(float)
df = df.dropna(subset=["size"])
df["status"] = df["status"].astype(int)
# przedstawiamy godzinę jako cykl aby odległość między 23 a 0 wynosiła 1 a nie 23
df["hour_sin"] = np.sin(2 * np.pi * df['hour'] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df['hour'] / 24)

features = ["status", "size", "hour_sin", "hour_cos", "req_GET", "req_POST", "req_HEAD","req_OPTIONS"]

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

model = IsolationForest(contamination=0.01, random_state=42)
df["anomaly"] = model.fit_predict(X)

medians = df[features].median()

def explain_anomaly(row):
    reasons = []
    if row["status"] not in [200, 304]:
        reasons.append(f"nietypowy kod statusu: {row['status']}")
    if row["size"] > medians["size"] * 3:
        reasons.append("bardzo duży rozmiar odpowiedzi")
    if row["method"] not in ["GET", "POST"]:
        reasons.append(f"nietypowa metoda: {row['method']}")
    return ", ".join(reasons) if reasons else "brak oczywistych przyczyn"

df["reasons"] = df.apply(lambda r: explain_anomaly(r) if r["anomaly"] == -1 else "",axis=1)

print("Liczba analizowanych logów: ", len(df))
print("\nLiczba wykrytych anomalii: ", len(df[df["anomaly"]==-1]))

print("\n\nLiczba wystąpień anomali ze względu na rodzaj anomalii:")
print(df["reasons"].value_counts())

print("\nMediana rozmiaru odpowiedzi: ", df["size"].median())
print("Przykładowe anomalie ze względu na rozmiar odpowiedzi:")
print(df[df["reasons"]=="bardzo duży rozmiar odpowiedzi"][["ip","size"]].head())

plt.figure(figsize=(12, 6))
plt.plot(requests_per_hour.index, requests_per_hour.values, marker='o')
plt.title("Liczba żądań na godzinę")
plt.xlabel("Godzina")
plt.ylabel("Liczba żądań")
plt.xticks(range(0, 24))
plt.grid()
plt.show()

plt.title("Rozmiar odpowiedzi w zależności od godziny z zaznaczonymi anomaliami")
plt.scatter(df["hour"], df["size"], c='blue', label='Normalne')
plt.scatter(df[df["anomaly"] == -1]["hour"], df[df["anomaly"] == -1]["size"], c='red', label='Anomalie')
plt.yscale('log')
plt.xlabel("Godzina")
plt.ylabel("Rozmiar odpowiedzi (log scale)")
plt.legend()
plt.grid()
plt.show()

plt.title("Wykres rozmiaru odpowiedzi z zaznaczonymi anomaliami")
plt.scatter(df.index, df["size"], c='blue', label='Normalne')
plt.scatter(df[df["anomaly"] == -1].index, df[df["anomaly"] == -1]["size"], c='red', label='Anomalie')
plt.yscale('log')
plt.xlabel("Indeks logu")
plt.ylabel("Rozmiar odpowiedzi (log scale)")
plt.legend()
plt.grid()
plt.show()