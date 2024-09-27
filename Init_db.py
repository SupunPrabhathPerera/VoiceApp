import sqlite3

DATABASE = 'app_database.db'

def create_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        dob TEXT NOT NULL,
        account_number TEXT UNIQUE NOT NULL,
        mfcc_features BLOB NOT NULL,
        balance REAL NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_database()

