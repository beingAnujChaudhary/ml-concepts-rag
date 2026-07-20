import psycopg
import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")
DB_NAME = os.getenv("DB_NAME", "rag_monitoring")
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    try:
        if DATABASE_URL:
            conn = psycopg.connect(DATABASE_URL)
        else:
            conn = psycopg.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                dbname=DB_NAME
            )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn is None:
        return
        
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_query TEXT,
                    rewritten_query TEXT,
                    answer TEXT,
                    response_time_ms INTEGER,
                    feedback INTEGER DEFAULT 0 -- 1 for positive, -1 for negative
                )
            """)
            conn.commit()
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

def log_interaction(user_query, rewritten_query, answer, response_time_ms):
    conn = get_db_connection()
    if conn is None:
        return None
        
    interaction_id = None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO interactions (user_query, rewritten_query, answer, response_time_ms)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (user_query, rewritten_query, answer, response_time_ms))
            interaction_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        print(f"Error logging interaction: {e}")
    finally:
        conn.close()
        
    return interaction_id

def log_feedback(interaction_id, feedback_value):
    conn = get_db_connection()
    if conn is None:
        return
        
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE interactions
                SET feedback = %s
                WHERE id = %s
            """, (feedback_value, interaction_id))
            conn.commit()
    except Exception as e:
        print(f"Error logging feedback: {e}")
    finally:
        conn.close()
