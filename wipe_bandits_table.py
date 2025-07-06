import sqlite3
import os

# Path to your database file
db_path = "./sol_prices.db"

# Make sure the file exists before attempting anything
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("DROP TABLE IF EXISTS bandit_logs;")
        conn.commit()
        print("✅ 'bandit_logs' table deleted successfully.")
    except sqlite3.Error as e:
        print(f"❌ Error deleting table: {e}")
    finally:
        conn.close()
else:
    print("❌ Database file not found.")
