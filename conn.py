from sqlalchemy import create_engine,text
import psycopg2
from sqlalchemy import Column, Integer, String, DateTime
import os

from dotenv import load_dotenv
load_dotenv()

# Access the variables
REAL_DB_USER = os.getenv("REAL_DB_USER")
REAL_DB_PASSWORD = os.getenv("REAL_DB_PASSWORD")
REAL_DB_HOST = os.getenv("REAL_DB_HOST")
REAL_DB_NAME = os.getenv("REAL_DB_NAME")
REAL_DB_PORT = os.getenv("REAL_DB_PORT")

engine =create_engine(
    # f"postgresql://{REAL_DB_USER}@{REAL_DB_HOST}:{REAL_DB_PORT}/{REAL_DB_NAME}"
    f"postgresql+psycopg2://{REAL_DB_USER}@{REAL_DB_HOST}:{REAL_DB_PORT}/{REAL_DB_NAME}"
)
def execute_query(query, params=None):
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            result = connection.execute(text(query), params)
            transaction.commit()
            return result
            # return result.fetchall() if result.returns_rows else None
    except Exception as e:
        print(f"An error occurred: {e}")

def insert_query(doc, embedding):
        try:
            # Insert query using parameterized format
            query = """
                INSERT INTO hr_helpdesk (description, description_embedding)
                VALUES (:doc, :embedding)
            """
            params = {'doc': doc, 'embedding': embedding}  # Use dictionary to pass params
            execute_query(query,params)
        except Exception as e:
            print(f"Insert query error: {e}")


def create_table(name):
    try:
        query=f"""
            CREATE TABLE IF NOT EXISTS {name} (
                description TEXT NOT NULL,
                description_embedding FLOAT8[] NOT NULL
            )
        """
        execute_query(query)
    except Exception as e:
        print("create table exception : ", e)

def fetch_chunks_and_embedding():
    try:
        query="SELECT description,description_embedding FROM hr_helpdesk"
        res = execute_query(query)
        # rows = res.fetchall()
        rows = res.mappings().all()
        description = [row['description'] for row in rows]
        embeddings = [row['description_embedding'] for row in rows]
        # print(embeddings)
        return description, embeddings
    except Exception as e:
        print("fetching failed : ", e)
        return None, None

fetch_chunks_and_embedding()