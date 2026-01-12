# db/connection.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("PG_DB", "fitness_ai"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASS", "jaffar@123"),
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432")
    )

def create_tables():
    """Create tables if they do not exist"""
    conn = get_connection()
    cur = conn.cursor()
    with open("db/schema.sql", "r") as f:
        cur.execute(f.read())
    conn.commit()
    conn.close()
    print("📌 Tables verified/created successfully")
