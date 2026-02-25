import psycopg2
from psycopg2 import sql


def get_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="aurex_db",
        user="postgres",
        password="KIKE1"
    )
    return conn