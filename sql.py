import sqlite3

def alter_users_table():
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()
    
    try:
        # Start a transaction
        cursor.execute("BEGIN TRANSACTION;")
        
        # Create the new Users_new table with the additional column
        cursor.execute("""
        CREATE TABLE Users_new (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            profile_pic TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Copy data from the old Users table to the new Users_new table
        cursor.execute("""
        INSERT INTO Users_new (user_id, username, email, password, created_at)
        SELECT user_id, username, email, password, created_at
        FROM Users;
        """)
        
        # Drop the old Users table
        cursor.execute("DROP TABLE Users;")
        
        # Rename the new table to Users
        cursor.execute("ALTER TABLE Users_new RENAME TO Users;")
        
        # Commit the transaction
        conn.commit()
        print("Old Users table dropped, new Users table created with the additional column successfully.")
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
        conn.rollback()  # Rollback the transaction in case of error
    finally:
        conn.close()

# Run the function to alter the Users table
if __name__ == "__main__":
    alter_users_table()
