import numpy as np
import pandas as pd
import joblib

# =====================
# LOAD MODEL
# =====================
model_pasut = joblib.load("model_pasut_rf.pkl")
model_angin = joblib.load("model_angin_rf.pkl")


# =====================
# RULE BASED STATUS
# =====================
def status_keamanan(pasut, angin):
    if pasut >= 2.4 and angin >= 0.55:
        return "Bahaya"
    elif pasut >= 2.2 or angin >= 0.45:
        return "Waspada"
    else:
        return "Aman"


# =====================
# FORECAST 24 JAM
# =====================
def forecast_for_date(df, target_date):

    features_pasut = [
        'Pasut_lag1','Pasut_lag2','Pasut_lag3','Pasut_lag6',
        'WS2M_lag1','WS2M_lag2',
        'u_lag1','v_lag1',
        'hour_sin','hour_cos'
    ]

    features_angin = [
        'WS2M_lag1','WS2M_lag2','WS2M_lag3',
        'u_lag1','v_lag1','u_lag2','v_lag2',
        'hour_sin','hour_cos'
    ]

    target_date = pd.Timestamp(target_date)

    seed = df[df["datetime"] < target_date].iloc[-1:].copy()
    seed["datetime"] = target_date - pd.Timedelta(hours=1)

    result = []

    for _ in range(24):
        pasut = model_pasut.predict(seed[features_pasut])[0]
        angin = model_angin.predict(seed[features_angin])[0]

        next_time = seed["datetime"].values[0] + pd.Timedelta(hours=1)
        hour = next_time.hour

        status = status_keamanan(pasut, angin)

        result.append({
            "Waktu": next_time,
            "Ketinggian Air (m)": round(pasut, 2),
            "Kecepatan Angin (km/jam)": round(angin * 3.6, 2),
            "Status": status
        })

        # update lag
        seed["Pasut_lag3"] = seed["Pasut_lag2"]
        seed["Pasut_lag2"] = seed["Pasut_lag1"]
        seed["Pasut_lag1"] = pasut

        seed["WS2M_lag3"] = seed["WS2M_lag2"]
        seed["WS2M_lag2"] = seed["WS2M_lag1"]
        seed["WS2M_lag1"] = angin

        seed["hour_sin"] = np.sin(2*np.pi*hour/24)
        seed["hour_cos"] = np.cos(2*np.pi*hour/24)
        seed["datetime"] = next_time

    return pd.DataFrame(result)
