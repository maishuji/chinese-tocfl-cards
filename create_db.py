import pandas as pd
import sqlite3

# Read the CSV file
df = pd.read_csv('translated_sentences.csv')
print(f"Loaded {len(df)} rows from CSV")

# Create SQLite database connection
conn = sqlite3.connect('chinese_flashcards.db')

# Write the dataframe to SQLite table
df.to_sql('sentences', conn, if_exists='replace', index=False)

print(f"Created table 'sentences' in database 'chinese_flashcards.db'")

# Verify the data
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM sentences")
count = cursor.fetchone()[0]
print(f"Verified: {count} rows in 'sentences' table")

# Show table schema
cursor.execute("PRAGMA table_info(sentences)")
columns = cursor.fetchall()
print(f"\nTable schema:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Show first few rows
print(f"\nFirst 3 rows:")
cursor.execute("SELECT * FROM sentences LIMIT 3")
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()
print("\nDatabase created successfully!")
