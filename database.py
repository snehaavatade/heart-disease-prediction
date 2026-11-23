
import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "heart_predictions.db"


def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            sex INTEGER,
            chest_pain INTEGER,
            cholesterol REAL,
            blood_pressure REAL,
            predicted_class INTEGER,
            probability REAL,
            prediction_date TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    age,
    sex,
    chest_pain,
    cholesterol,
    blood_pressure,
    predicted_class,
    probability
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            age,
            sex,
            chest_pain,
            cholesterol,
            blood_pressure,
            predicted_class,
            probability,
            prediction_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        age,
        sex,
        chest_pain,
        cholesterol,
        blood_pressure,
        predicted_class,
        probability,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_predictions():
    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()
    return df
