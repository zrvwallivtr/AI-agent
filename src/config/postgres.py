import os
import psycopg2
from dotenv import load_dotenv

from src.config.files_and_directories import ENV_PATH

load_dotenv(dotenv_path=ENV_PATH)

DBNAME = os.getenv("PGDB_DBNAME")
USER = os.getenv("PGDB_USER")
PASSWORD = os.getenv("PGDB_PASSWORD")
HOST = os.getenv("PGDB_HOST")
PORT = os.getenv("PGDB_PORT")


conn = psycopg2.connect(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)
