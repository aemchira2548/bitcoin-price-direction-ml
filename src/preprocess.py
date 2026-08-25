"""
Data Preprocessing - Bitcoin Price Direction Prediction
=========================================================
โหลดข้อมูลราคา Bitcoin รายวัน (2019-2024) แล้วทำความสะอาด + สร้าง feature
สำหรับโมเดล Machine Learning ที่ทำนายว่าราคาพรุ่งนี้จะ "ขึ้น" หรือ "ลง"
"""

import pandas as pd
import numpy as np


def clean_numeric(series, is_percent=False, is_volume=False):
    """แปลง string เช่น '60,309.1' หรือ '106.30K' หรือ '3.60%' ให้เป็นตัวเลข"""
    s = series.astype(str).str.replace(",", "", regex=False)

    if is_percent:
        s = s.str.replace("%", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    if is_volume:
        def parse_vol(x):
            x = x.strip()
            if x in ("", "nan", "-"):
                return np.nan
            multiplier = 1
            if x.endswith("K"):
                multiplier = 1_000
                x = x[:-1]
            elif x.endswith("M"):
                multiplier = 1_000_000
                x = x[:-1]
            elif x.endswith("B"):
                multiplier = 1_000_000_000
                x = x[:-1]
            try:
                return float(x) * multiplier
            except ValueError:
                return np.nan
        return s.apply(parse_vol)

    return pd.to_numeric(s, errors="coerce")


def load_and_clean(raw_path):
    df = pd.read_csv(raw_path)

    # 1) แปลงคอลัมน์วันที่ และเรียงจากเก่า -> ใหม่ (ไฟล์ดิบเรียงใหม่ -> เก่า)
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y")
    df = df.sort_values("Date").reset_index(drop=True)

    # 2) ทำความสะอาดคอลัมน์ตัวเลข (มี comma, หน่วย K/M, % ปนอยู่ในข้อความ)
    df["Close"] = clean_numeric(df["Price"])
    df["Open"] = clean_numeric(df["Open"])
    df["High"] = clean_numeric(df["High"])
    df["Low"] = clean_numeric(df["Low"])
    df["Volume"] = clean_numeric(df["Vol."], is_volume=True)
    df["Change_Pct"] = clean_numeric(df["Change %"], is_percent=True)

    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Change_Pct"]]

    # 3) จัดการค่าว่าง (Volume บางวันไม่มีข้อมูล) ด้วยการเติมค่ามัธยฐาน
    df["Volume"] = df["Volume"].fillna(df["Volume"].median())
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    # 4) Feature Engineering
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"] * 100

    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()
    df["MA_Ratio"] = df["MA_7"] / df["MA_21"]

    df["Volatility_7"] = df["Daily_Return"].rolling(window=7).std()

    # RSI (Relative Strength Index) 14 วัน
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["Volume_Change"] = df["Volume"].pct_change() * 100

    # 5) Target: พรุ่งนี้ราคาปิดสูงกว่าวันนี้หรือไม่ (1 = ขึ้น, 0 = ลง)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # ตัดแถวที่มี NaN จาก rolling window และแถวสุดท้ายที่ไม่มี target
    df_model = df.dropna().reset_index(drop=True)

    return df, df_model


if __name__ == "__main__":
    raw_df, model_df = load_and_clean("data/bitcoin_raw.csv")
    raw_df.to_csv("data/bitcoin_clean.csv", index=False)
    model_df.to_csv("data/bitcoin_features.csv", index=False)

    print("ข้อมูลดิบหลังทำความสะอาด:", raw_df.shape)
    print("ข้อมูลพร้อมสำหรับโมเดล (มี feature ครบ):", model_df.shape)
    print("\nสัดส่วน Target (0=ลง, 1=ขึ้น):")
    print(model_df["Target"].value_counts(normalize=True).round(3))
    print("\nตัวอย่างข้อมูล:")
    print(model_df.head(3).to_string())
