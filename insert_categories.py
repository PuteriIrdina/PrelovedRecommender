import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

# Define the SQL statement for insertion
insert_query = """
INSERT INTO Categories (category_id, category_name) VALUES
(1, 'Books'),
(2, 'Clothing'),
(3, 'Furniture'),
(4, 'Kitchenware and appliances'),
(5, 'Home Decor'),
(6, 'Electronics'),
(7, 'Toys and Games'),
(8, 'Sports Equipment'),
(9, 'Bedding and linens'),
(10, 'Personal Care Products');
"""

try:
    # Execute the SQL statement
    cursor.executescript(insert_query)
    conn.commit()
    print("Insertion successful")
except Exception as e:
    print(f"Error: {e}")

# Close the database connection
conn.close()
