# data_handling.py
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

def load_db_credentials():
    """
    Load database credentials from the .env file.
    """
    load_dotenv()
    db_config = {
        'host': os.getenv("DB_HOST"),
        'dbname': os.getenv("DB_NAME"),
        'user': os.getenv("DB_USER"),
        'password': os.getenv("DB_PASSWORD")
    }
    return db_config

def connect_to_database():
    """
    Establish a connection to the SQL database using credentials from .env.
    """
    try:
        db_config = load_db_credentials()
        conn = psycopg2.connect(**db_config)
        print("Database connection established.")
        return conn
    except Exception as e:
        print("Error connecting to the database:", e)
        return None

def fetch_data_from_query(conn, query):
    """
    Fetch data using a SQL query and return it as a pandas DataFrame.
    """
    try:
        df = pd.read_sql_query(query, conn)
        print("Data fetched successfully.")
        return df
    except Exception as e:
        print("Error fetching data from query:", e)
        return None

def close_database_connection(conn):
    """
    Close the database connection.
    """
    try:
        conn.close()
        print("Database connection closed.")
    except Exception as e:
        print("Error closing the database connection:", e)
