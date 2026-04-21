import sqlite3
import os

DB_FILE = "traffic_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analytics_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_vehicles INTEGER,
            vpm REAL,
            car_count INTEGER,
            bike_count INTEGER,
            truck_count INTEGER,
            bus_count INTEGER,
            risk_pct REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_stats(stats, risk):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # We expect stats to have 'total', 'vpm', 'types' dictionary
    total = stats.get("total", 0)
    vpm = stats.get("vpm", 0.0)
    types = stats.get("types", {})
    cars = types.get("Car", 0)
    bikes = types.get("Bike", 0)
    trucks = types.get("Truck", 0)
    buses = types.get("Bus", 0)
    
    c.execute('''
        INSERT INTO analytics_logs 
        (total_vehicles, vpm, car_count, bike_count, truck_count, bus_count, risk_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (total, vpm, cars, bikes, trucks, buses, float(risk)))
    
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM analytics_logs ORDER BY timestamp DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    return rows
