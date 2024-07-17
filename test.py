import sqlite3
import random
import string
from datetime import datetime, timedelta
import faker

# Initialize Faker
fake = faker.Faker()

# Categories and conditions
categories = ["Books", "Clothing", "Furniture", "Kitchenware and appliances", "Home Decor", "Electronics", "Toys and Games", "Sports Equipment", "Bedding and linens", "Personal Care Products"]
conditions = ["Brand new", "Like new", "Lightly used", "Used", "Well used"]

# Generate random usernames, emails, and passwords
def generate_random_user():
    username = fake.user_name()
    email = fake.email()
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    created_at = fake.date_time_this_year(before_now=True, after_now=False)
    return (username, email, password, created_at)

# Generate random preference data
def generate_random_preference(user_id):
    category = random.choice(categories)
    condition = random.choice(conditions)
    min_price = random.randint(10, 100)
    max_price = random.randint(min_price + 1, min_price + 100)
    created_at = fake.date_time_this_year(before_now=True, after_now=False)
    return (user_id, category, condition, min_price, max_price, created_at)

# Database connection
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

# Insert users
for _ in range(47):
    user = generate_random_user()
    cursor.execute("INSERT INTO Users (username, email, password, created_at) VALUES (?, ?, ?, ?)", user)
    user_id = cursor.lastrowid  # Get the user_id of the newly inserted user
    
    # Insert preference for the user
    preference = generate_random_preference(user_id)
    cursor.execute("INSERT INTO Preferences (user_id, category_id, condition, min_price, max_price, created_at) VALUES (?, ?, ?, ?, ?, ?)", preference)

# Commit the transaction
conn.commit()

# Close the connection
conn.close()
