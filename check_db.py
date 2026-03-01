import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('chinese_flashcards.db')

# Get all tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [table[0] for table in cursor.fetchall()]

print(f"Database: chinese_flashcards.db")
print(f"Tables: {', '.join(tables)}\n")
print("=" * 80)

# Display head of each table
for table_name in tables:
    print(f"\nTable: {table_name}")
    print("-" * 80)
    
    # Get table info
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"Columns: {', '.join([col[1] for col in columns])}")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"Total rows: {count}")
    
    # Read first 5 rows using pandas for better formatting
    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
    print(f"\nFirst 5 rows:")
    print(df.to_string(index=False))
    print("\n" + "=" * 80)

conn.close()
