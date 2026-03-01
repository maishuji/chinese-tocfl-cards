import pandas as pd
import sqlite3

# Read the words CSV file
df = pd.read_csv('data/words_final.csv')
print(f"Loaded {len(df)} rows from words_final.csv")
print(f"Columns: {list(df.columns)}")

# Create SQLite database connection
conn = sqlite3.connect('chinese_flashcards.db')

# Write the dataframe to SQLite table
df.to_sql('words', conn, if_exists='replace', index=False)

print(f"\nCreated table 'words' in database 'chinese_flashcards.db'")

# Verify the data
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM words")
count = cursor.fetchone()[0]
print(f"Verified: {count} rows in 'words' table")

# Show table schema
cursor.execute("PRAGMA table_info(words)")
columns = cursor.fetchall()
print(f"\nTable schema:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Show first few rows
print(f"\nFirst 3 rows:")
cursor.execute("SELECT * FROM words LIMIT 3")
for row in cursor.fetchall():
    print(f"  {row}")

# Show both tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"\nAll tables in database:")
for table in tables:
    print(f"  - {table[0]}")

conn.close()
print("\nWords table added successfully!")
