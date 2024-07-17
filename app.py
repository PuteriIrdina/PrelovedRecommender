import streamlit as st
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from PIL import Image
from sqlalchemy import DECIMAL, create_engine, MetaData, Table, Column, Integer, String, TIMESTAMP, ForeignKey, insert, select, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import insert
import datetime
import pandas as pd
import numpy as np
import pickle
import sqlite3
import os
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page configuration
st.set_page_config(page_title="My Preloved Item Recommendation", page_icon="😁", layout="wide")

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_type' not in st.session_state:
    st.session_state.user_type = None

# Database configuration
DATABASE_URI = "sqlite:///my_database.db"
engine = create_engine(DATABASE_URI)
metadata = MetaData()
metadata.bind = engine
Session = sessionmaker(bind=engine)
session = Session()

# Define the users table
users = Table('Users', metadata,
              Column('user_id', Integer, primary_key=True, autoincrement=True),
              Column('username', String(50), nullable=False),
              Column('email', String(100), nullable=False, unique=True),
              Column('password', String(255), nullable=False),
              Column('profile_pic', String(255)),  # Add this line
              Column('created_at', TIMESTAMP, default=datetime.datetime.utcnow)
              )

# Define the categories table
categories = Table('categories', metadata,
                   Column('category_id', Integer, primary_key=True),
                   Column('category_name', String(100), nullable=False)
                   )

# Define Preferences table
preferences = Table('Preferences', metadata,
                    Column('preference_id', Integer, primary_key=True, autoincrement=True),
                    Column('user_id', Integer, ForeignKey('Users.user_id')),
                    Column('category_id', Integer, ForeignKey('categories.category_id')),
                    Column('condition', String(50)),
                    Column('min_price', Integer),
                    Column('max_price', Integer),
                    Column('created_at', TIMESTAMP, default=datetime.datetime.utcnow)
                    )

# Define Interactions table
interactions = Table('Interactions', metadata,
                     Column('interaction_id', Integer, primary_key=True, autoincrement=True),
                     Column('user_id', Integer, ForeignKey('Users.user_id')),
                     Column('product_id', Integer, ForeignKey('Products.product_id')),
                     Column('interaction_type', String(50)),
                     Column('interaction_time', TIMESTAMP, default=datetime.datetime.utcnow)
                     )

# Define Sellers table
sellers = Table('Sellers', metadata,
                Column('seller_id', Integer, primary_key=True, autoincrement=True),
                Column('username', String(50), nullable=False),
                Column('email', String(100), nullable=False, unique=True),
                Column('password', String(255), nullable=False),  # Ensure this line is present
                Column('contact_number', String(20)),
                Column('created_at', TIMESTAMP, default=datetime.datetime.utcnow)
                )

# Define Products table
products = Table('Products', metadata,
                 Column('product_id', Integer, primary_key=True, autoincrement=True),
                 Column('product_name', String(255), nullable=False),
                 Column('product_condition', String(50)),
                 Column('product_price', DECIMAL(10, 2), nullable=False),
                 Column('product_stock', Integer, nullable=False),
                 Column('product_image', String(255)),
                 Column('category_id', Integer, ForeignKey('categories.category_id')),
                 Column('seller_id', Integer, ForeignKey('Sellers.seller_id'))
                 )

metadata.create_all(engine)  # Create tables

#---------------RECOMMENDER------------------------
# Load your pre-trained Word2Vec model
word2vec_model = gensim.models.Word2Vec.load('/workspaces/PrelovedRecommender/word2vec_model.bin')

# Load your pre-trained TF-IDF vectorizer
with open('/workspaces/PrelovedRecommender/tfidf_vectorizer_text.pkl', 'rb') as f:
    tfidf_vectorizer_text = pickle.load(f)

# Load your cleaned dataset
df_cleaned = pd.read_csv('/workspaces/PrelovedRecommender/Dataset_products(new).csv', encoding='latin1')

# If additional cleaning or preprocessing is needed
df_cleaned['Product_name'] = df_cleaned['Product_name'].astype(str)
df_cleaned['Product_condition'] = df_cleaned['Product_condition'].astype(str)
df_cleaned['Category_id'] = df_cleaned['Category_id'].astype(str)

def get_word_embeddings(text):
    words = text.split()
    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def recommend_based_on_combined_tfidf(product_name, category_id=None, price_range=None, condition=None, top_n=10):
    sentences = df_cleaned['Product_condition'].apply(lambda x: x.split()) + df_cleaned['Category_id'].apply(lambda x: x.split())
    X_text_tfidf = tfidf_vectorizer_text.transform(df_cleaned['Product_name'])
    X_meta_embeddings = np.array([get_word_embeddings(condition + ' ' + category) for condition, category in zip(df_cleaned['Product_condition'], df_cleaned['Category_id'])])
    X_combined_tfidf = sp.hstack((X_text_tfidf, sp.csr_matrix(X_meta_embeddings)))

    filtered_df = df_cleaned
    if category_id:
        filtered_df = filtered_df[filtered_df['Category_id'] == category_id]
    if price_range:
        filtered_df = filtered_df[
            (filtered_df['Product_price'] >= price_range[0]) &
            (filtered_df['Product_price'] <= price_range[1])
        ]
    if condition:
        filtered_df = filtered_df[filtered_df['Product_condition'] == condition]

    if product_name not in filtered_df['Product_name'].values:
        return []

    product_index = df_cleaned[df_cleaned['Product_name'] == product_name].index[0]
    cos_similarities = cosine_similarity(X_combined_tfidf[product_index], X_combined_tfidf).flatten()
    top_indices = cos_similarities.argsort()[-top_n-1:-1]

    recommended_products = df_cleaned.iloc[top_indices][['Product_id', 'Product_name', 'Product_image', 'Product_price', 'Product_condition']].to_dict(orient='records')
    return recommended_products

# Update sidebar function
def update_sidebar():
    if st.session_state.logged_in:
        if st.session_state.user_type == "Customer":
            with st.sidebar:
                selected = option_menu(
                    menu_title="Menu",
                    options=["Customer page","Profile", "Recommendation", "Search", "List", "Logout"],  # Customer menu options
                    icons=["house","person-fill", "bag", "search", "database", "door-closed"],
                    menu_icon="cast",
                    default_index=0,
                )
        elif st.session_state.user_type == "Seller":
            with st.sidebar:
                selected = option_menu(
                    menu_title="Menu",
                    options=["Seller page", "Manage Products", "Seller List", "Logout"],  # Seller menu options
                    icons=["house", "box-seam","database", "door-closed"],
                    menu_icon="cast",
                    default_index=0,
                )
    else:
        with st.sidebar:
            selected = option_menu(
                menu_title="Menu",
                options=["Homepage", "Register", "Login"],
                icons=["house", "pencil-square", "door-open-fill"],
                menu_icon="cast",
                default_index=0,
            )
    return selected

# Main function
def main():
    selected = update_sidebar()  # Capture the returned value

    if st.session_state.logged_in:
        user_id = st.session_state.user_id 
        
        if st.session_state.user_type == "Customer":
            if selected == "Customer page":
                st.title(f"You are at {selected}")
                user_page()
            elif selected == "Profile":
                st.title(f"Profile")
                customer_profile(user_id)
            elif selected == "Recommendation" and st.session_state.user_type == "Customer":
                st.title(f"You are at {selected} page")
                recommend_page()
            elif selected == "Search":
                st.title(f"Search Products")
                search_page()
            elif selected == "List":
                st.title(f"List of Users and Preferences")
                list()
            elif selected == "Logout":
                st.title(f"You are at {selected} page")
                logout()
        elif st.session_state.user_type == "Seller":
            if selected == "Seller page":
                st.title(f"You are at {selected}")
                seller_page()
            elif selected == "Manage Products" and st.session_state.user_type == "Seller":
                st.title(f"Manage Your Products")
                manage_products_page()
            elif selected == "Seller List":
                st.title(f"List of Users and Preferences")
                seller_list()
            elif selected == "Logout":
                st.title(f"You are at {selected} page")
                logout()
    else:
        if selected == "Homepage":
            st.title(f"You are at {selected}")
            homepage()
        elif selected == "Register":
            st.title(f"You are at {selected} page")
            choose_registration()
        elif selected == "Login":
            st.title(f"You are at {selected} page")
            login_page()

#-----------------HOEMPAGE-----------------
def homepage():
    # Header section
    with st.container():
        st.title("Welcome to Preloved Item Recommender!")
        st.subheader("Discover Your Perfect Preloved Items")
        st.write("At Preloved Item Recommender, we believe in giving items a second life. By choosing preloved products, you not only save money but also contribute to a more sustainable future. Join us in making a difference, one item at a time.")

    # Main page
    with st.container():
        st.write("----")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("How It Works")
            st.write("1. **Register**: Create an account and tell us about your preferences.")
            st.write("2. **Discover**: Browse through personalized recommendations of preloved items.")
            st.write("3. **Enjoy**: Purchase items you love and feel good about reducing waste.")
            st.write("[YouTube Channel >](https://www.youtube.com/watch?v=Pe4gUSIxte8)")
        with right_column:
            lottie_coding = load_lottieurl("https://lottie.host/37713d2b-ffd6-48db-b024-ffc348d32248/C1J6R7FTKt.json")
            st_lottie(lottie_coding, height=300, key="coding")

    with st.container():
        st.header("Environmental Benefits")
        image_column, text_column = st.columns((1, 2))

        with image_column:
            img_contact_form = Image.open("images/landfills.jpg")
            st.image(img_contact_form)
            img_contact_form = Image.open("images/landfills2.jpg")
            st.image(img_contact_form)

        with text_column:
            st.header("Why Preloved?")
            st.write("By purchasing preloved items, you're playing a crucial role in reducing the amount of waste that ends up in landfills. Every item reused means less demand for new products, conserving resources and minimizing pollution.")
            st.write("Here are some key benefits:")
            st.write("- **Waste Reduction**: Keeping items out of landfills helps reduce environmental pollution.")
            st.write("- **Resource Conservation**: Reusing items means fewer resources are needed to produce new ones.")
            st.write("- **Carbon Footprint**: Lower demand for new products means less energy consumption and reduced carbon emissions.")
            st.write("Join us in our mission to promote sustainability and make a positive impact on the environment.")

    # Define the footer content
    st.markdown("""
    ---
    #### Contact Us
    - **Email:** contact@ecoshop.com
    - **Phone:** +1-123-456-7890
    
    #### Follow Us
    [![Twitter](https://img.shields.io/twitter/follow/ecoshop?style=social)](https://twitter.com/ecoshop)
    [![Instagram](https://img.shields.io/badge/Instagram-ecoshop-9cf?style=social&logo=instagram)](https://instagram.com/ecoshop)
    
    #### Visit Us
    [EcoShop Website](https://ecoshop.com)
    
    #### Legal
    [Privacy Policy](https://ecoshop.com/privacy)
    [Terms of Service](https://ecoshop.com/terms)
        """)

# Load Lottie animation
def load_lottieurl(url):
    import requests
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#------------REGISTER PAGE------------------
def choose_registration():
    st.subheader("Choose Registration Type")
    user_type = st.radio("Register as:", ('Customer', 'Seller'))
    if user_type == 'Customer':
        register_customer()
    else:
        register_seller()

#------CUSTOMER REGISTER------
def register_customer():
    st.subheader("Customer Registration")
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        category_name = st.selectbox("Category Preference", ["Books", "Clothing", "Furniture", "Kitchenware and appliances",
                                                             "Home Decor", "Electronics", "Toys and Games",
                                                             "Sports Equipment", "Bedding and linens", "Personal Care Products"])
        condition = st.selectbox("Product Condition Preference", ["Brand new", "Like new", "Lightly used", "Used", "Well used"])
        min_price = st.slider("Minimum Price Preference", min_value=0, max_value=1000, step=10)
        max_price = st.slider("Maximum Price Preference", min_value=0, max_value=1000, step=10)
        submit = st.form_submit_button("Register")

        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                # Insert user into Users table
                new_user = users.insert().values(username=username, email=email, password=password, created_at=datetime.datetime.utcnow())
                result = session.execute(new_user)
                user_id = result.inserted_primary_key[0]  # Get the inserted user_id
                session.commit()

                # Get category_id from Categories table based on category_name
                query_category = categories.select().where(categories.c.category_name == category_name)
                result_category = session.execute(query_category).fetchone()
                if result_category:
                    # Ensure to use positional index to access the column value
                    category_id = result_category[0]

                    # Insert user preferences into Preferences table
                    new_preference = preferences.insert().values(user_id=user_id, category_id=category_id, condition=condition, min_price=min_price, max_price=max_price, created_at=datetime.datetime.utcnow())
                    session.execute(new_preference)
                    session.commit()

                    st.success("Registration successful!")
                else:
                    st.error("Failed to retrieve category information")
#------SELLER REGISTER------
def register_seller():
    st.subheader("Seller Registration")
    with st.form("register_seller_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        contact_number = st.text_input("Contact Number")
        submit = st.form_submit_button("Register")

        if submit:
            try:
                # Insert seller into Sellers table
                new_seller = sellers.insert().values(
                    username=username,
                    email=email,
                    password=password,
                    contact_number=contact_number,
                    created_at=datetime.datetime.utcnow()
                )
                result = session.execute(new_seller)
                session.commit()
                st.success("Seller registered successfully!")
            except Exception as e:
                session.rollback()
                st.error(f"An error occurred: {e}")

#-------------LOGIN PAGE ------------------
def login_page():
    st.write("Please enter your login credentials.")

    with st.form(key="login_form", clear_on_submit=True):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")

        if submit_button:
            if email and password:
                # Perform login validation here
                if login_is_successful(email, password):
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.warning("Invalid email or password. Please try again.")
            else:
                st.warning("Please fill in all fields.")

# Function to validate login
def login_is_successful(email, password):
    # Replace with your actual login validation logic
    with engine.begin() as connection:
        # Check if user is a customer
        customer_result = connection.execute(users.select().where(users.c.email == email, users.c.password == password)).fetchone()
        # Check if user is a seller
        seller_result = connection.execute(sellers.select().where(sellers.c.email == email, sellers.c.password == password)).fetchone()

        if customer_result or seller_result:
            if customer_result:
                st.session_state.user_id = customer_result[0]
                st.session_state.user_type = "Customer"
            elif seller_result:
                st.session_state.user_id = seller_result[0]
                st.session_state.user_type = "Seller"
            return True
        else:
            return False

#---------LOGOUT BUTTON-----------                
def logout():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_type = None
    st.success("Logged out successfully.")
    st.rerun()

#------------USER FRONT PAGE--------------
# Function to check if the user is logged in
def is_user_logged_in():
    return "user_id" in st.session_state

# Function to display the user page
def user_page():
    # Header section
    with st.container():
         st.title(f"Welcome Back")
         st.subheader("Your Personalized Dashboard")
         st.write("Here, you can manage your profile, view your preferences, and explore personalized recommendations.")

#------------CUSTOMER PROFILE--------------
def get_customer_data(user_id):
    with engine.connect() as conn:
        query = select(
            users.c.user_id,
            users.c.username,
            users.c.email,
            users.c.password,
            users.c.profile_pic,
            users.c.created_at
        ).where(users.c.user_id == user_id)
        result = conn.execute(query).fetchone()
        if result:
            customer_data = {
                'user_id': result[0],
                'username': result[1],
                'email': result[2],
                'password': result[3],
                'profile_pic': result[4],
                'created_at': result[5]
            }
            return customer_data
        return None

# Define the update_customer_data function
def update_customer_data(user_id, username, email, password=None, profile_pic=None):
    with engine.connect() as conn:
        stmt = (
            update(users)
            .where(users.c.user_id == user_id)
            .values(username=username, email=email, password=password, profile_pic=profile_pic)
        )
        conn.execute(stmt)

# Define the customer_profile function
def customer_profile(user_id):
    st.title("Customer Profile")
    
    customer_data = get_customer_data(user_id)
    
    if customer_data["profile_pic"]:
        st.image(customer_data["profile_pic"], width=150)
    
    with st.form(key='profile_form'):
        new_username = st.text_input("Username", customer_data["username"])
        new_email = st.text_input("Email", customer_data["email"])
        new_password = st.text_input("Password", type='password')
        new_profile_pic = st.file_uploader("Profile Picture", type=['jpg', 'png'])
        
        submit_button = st.form_submit_button(label='Update Profile')
        
        if submit_button:
            # Update logic here
            profile_pic_path = customer_data["profile_pic"]
            if new_profile_pic:
                # Step 2: Ensure the directory exists
                profile_pic_dir = "profile_pics"
                os.makedirs(profile_pic_dir, exist_ok=True)
                
                # Step 3: Save the profile picture
                profile_pic_path = os.path.join(profile_pic_dir, new_profile_pic.name)
                with open(profile_pic_path, "wb") as f:
                    f.write(new_profile_pic.getbuffer())

            update_customer_data(user_id, new_username, new_email, new_password or customer_data["password"], profile_pic_path)

            st.success("Profile updated successfully!")

#------------ RECOMMEND PAGE---------------
# Function to log interaction using SQLAlchemy
def log_interaction(user_id, product_id):
    # Define interaction type
    interaction_type = "click on product"

    # Create a new interaction record
    interaction_values = {
        'user_id': user_id,
        'product_id': product_id,
        'interaction_type': interaction_type,
        'interaction_time': datetime.datetime.utcnow()
    }

    try:
        # Execute insertion
        with engine.connect() as conn:
            new_interaction = interactions.insert().values(**interaction_values)
            result = conn.execute(new_interaction)
            conn.commit()  # Ensure the transaction is committed

        # Log success or failure
        if result.rowcount == 1:
            print(f"Interaction logged successfully: User {user_id} clicked on product {product_id}.")
        else:
            print("Failed to log interaction.")
    except Exception as e:
        print(f"Error logging interaction: {e}")

# Function to handle product click
def handle_product_click(user_id, product):
    log_interaction(user_id, product['Product_id'])
    st.session_state.view_product = product
    
def display_product(product):
    st.markdown(f"""
    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center; width: 300px; margin: auto;">
        <img src="{product['Product_image']}" style="width: 200px; height: 200px; object-fit: cover; border-radius: 5px;">
        <h3 style="margin: 10px 0; height: 50px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{product['Product_name']}</h3>
        <p>Price: ${product['Product_price']}</p>
        <p>Condition: {product['Product_condition']}</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Back to Recommendations"):
        st.session_state.view_product = None

#------------ RECOMMEND PAGE---------------
def display_product(product):
    st.markdown(f"""
    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center; width: 300px; margin: auto;">
        <img src="{product['Product_image']}" style="width: 200px; height: 200px; object-fit: cover; border-radius: 5px;">
        <h3 style="margin: 10px 0; height: 50px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{product['Product_name']}</h3>
        <p>Price: ${product['Product_price']}</p>
        <p>Condition: {product['Product_condition']}</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Back to Recommendations"):
        st.session_state.view_product = None

def recommend_page():
    if st.session_state.get('logged_in', False):
        st.header("Recommended Items")

        # Fetch user_id from session_state
        user_id = st.session_state.user_id

        if user_id:
            query = preferences.select().where(preferences.c.user_id == user_id)
            result = session.execute(query).fetchone()
            
            if result:
                # Assuming result is a tuple or named tuple
                user_preferences = {
                    'category_id': result[2],  # Adjust indices as per your database schema
                    'condition': result[3],
                    'min_price': result[4],
                    'max_price': result[5]
                }

                # Load the dataset
                df = pd.read_csv('/workspaces/PrelovedRecommender/Dataset_products(new).csv', encoding='latin1')

                # Ensure 'Product_price' is numeric
                df['Product_price'] = pd.to_numeric(df['Product_price'], errors='coerce')
                df = df.dropna(subset=['Product_price'])

                # Convert user preferences prices to numeric
                user_preferences['min_price'] = pd.to_numeric(user_preferences['min_price'], errors='coerce')
                user_preferences['max_price'] = pd.to_numeric(user_preferences['max_price'], errors='coerce')

                # Filter the dataset based on user preferences
                filtered_df = df[
                    (df['Category_id'] == user_preferences['category_id']) &
                    (df['Product_condition'] == user_preferences['condition']) &
                    (df['Product_price'] >= user_preferences['min_price']) &
                    (df['Product_price'] <= user_preferences['max_price'])
                ]

                # Debug: Print the filter criteria
                st.write("Filter Criteria:")
                st.write(f"Category ID: {user_preferences['category_id']}")
                st.write(f"Condition: {user_preferences['condition']}")
                st.write(f"Min Price: {user_preferences['min_price']}")
                st.write(f"Max Price: {user_preferences['max_price']}")

                recommended_products = filtered_df.head(10)  # Assuming you want the top 10 recommendations

                if not recommended_products.empty:
                    st.write("Recommended Products:")

                    # Display products in rows of 3-4 items
                    if 'view_product' not in st.session_state:
                        st.session_state.view_product = None

                    if st.session_state.view_product is None:
                        cols_per_row = 4
                        num_products = len(recommended_products)
                        rows = num_products // cols_per_row + (num_products % cols_per_row > 0)

                        for i in range(rows):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                if i * cols_per_row + j < num_products:
                                    product = recommended_products.iloc[i * cols_per_row + j]
                                    with cols[j]:
                                        if st.button(
                                                "", 
                                               key=f"{i * cols_per_row + j}_button",
                                                on_click=handle_product_click,
                                                args=(user_id, product)):
                                            st.session_state.view_product = product

                                        st.markdown(f"""
                                        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center; height: 300px; width: 200px; display: flex; flex-direction: column; align-items: center; justify-content: space-between;">
                                            <img src="{product['Product_image']}" style="width: 150px; height: 150px; object-fit: cover; border-radius: 5px;">
                                            <div style="text-align: center; height: 60px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                                <h6 style="margin: 10px 0; font-size: 16px;">{product['Product_name']}</h5>
                                            </div>
                                            <p style="margin: 5px 0;">Price: RM{product['Product_price']}</p>
                                            <p style="margin: 5px 0;">Condition: {product['Product_condition']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)

                    else:
                        display_product(st.session_state.view_product)

                else:
                    st.write("No products match the current filter criteria.")
            else:
                st.error("No preferences found for the logged-in user")
        else:
            st.error("User ID not found in session")
    else:
        st.warning("Please login to access recommendations.")

#-------SEARCH Products------------

# Load product data from CSV
def load_data():
    return pd.read_csv('/workspaces/PrelovedRecommender/Dataset_products(new).csv', encoding='latin1')

product_data = load_data()

def search_page():
    search_term = st.text_input("Enter product name to search:")
    if search_term:
        results = product_data[product_data['Product_name'].str.contains(search_term, case=False)]
        if not results.empty:
            st.write("Search Results:")

            # Display products in rows of 3-4 items
            cols_per_row = 4
            num_products = len(results)
            rows = num_products // cols_per_row + (num_products % cols_per_row > 0)

            for i in range(rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i * cols_per_row + j < num_products:
                        product = results.iloc[i * cols_per_row + j]
                        with cols[j]:
                            if st.button(
                                    "",
                                    key=f"search_{i * cols_per_row + j}_button",
                                    on_click=handle_product_click,
                                    args=(st.session_state.user_id, product)):
                                st.session_state.view_product = product

                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center; height: 300px; width: 200px; display: flex; flex-direction: column; align-items: center; justify-content: space-between;">
                                <img src="{product['Product_image']}" style="width: 150px; height: 150px; object-fit: cover; border-radius: 5px;">
                                <div style="text-align: center; height: 60px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                    <h6 style="margin: 10px 0; font-size: 16px;">{product['Product_name']}</h6>
                                </div>
                                <p style="margin: 5px 0;">Price: RM{product['Product_price']}</p>
                                <p style="margin: 5px 0;">Condition: {product['Product_condition']}</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.write("No products found.")

# Ensure 'view_product' is in session state
if 'view_product' not in st.session_state:
    st.session_state.view_product = None

#-------LIST IN DATABASE FOR USERS------------
def list():    

    # Function to get users from the database
    def get_users():
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        select_query = "SELECT * FROM Users;"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return rows, columns

    # Function to get preferences from the database
    def get_preferences():
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        select_query = "SELECT * FROM Preferences;"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return rows, columns

    # Function to get categories from the database
    def get_categories():
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        select_query = "SELECT * FROM Categories;"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return rows, columns

    # Function to get interactions from the database
    def get_interactions():
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        select_query = "SELECT * FROM Interactions;"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return rows, columns

    # Streamlit app
    st.title("Database Tables")

    # Display Users table
    st.header("Users")
    st.write("List of Users from the database:")
    users, user_columns = get_users()
    user_df = pd.DataFrame(users, columns=user_columns)
    st.table(user_df)

    # Display Preference table
    st.header("Preference")
    st.write("List of Preferences from the database:")
    preferences, preference_columns = get_preferences()
    preference_df = pd.DataFrame(preferences, columns=preference_columns)
    st.table(preference_df)

    # Display Categories table
    st.header("Categories")
    st.write("List of Categories from the database:")
    categories, category_columns = get_categories()
    category_df = pd.DataFrame(categories, columns=category_columns)
    st.table(category_df)

    # Display Interactions table
    st.header("Interactions")
    st.write("List of Interactions from the database:")
    interactions, interaction_columns = get_interactions()
    interaction_df = pd.DataFrame(interactions, columns=interaction_columns)
    st.table(interaction_df)

#------------SELLER FRONT PAGE--------------
# Function to display the seller page
def seller_page():
    # Header section
    with st.container():
         st.title(f"Welcome Back")
         st.subheader("Your Personalized Dashboard")
         st.write("Here, you can manage your products.")

def manage_products_page():
    st.write("Manage your products here.")
    seller_id = st.session_state.user_id

    # Add product form
    with st.form("add_product_form"):
        product_name = st.text_input("Product Name")
        product_condition = st.selectbox("Condition", ["Brand new", "Like new", "Lightly used", "Used", "Well used"])
        product_price = st.number_input("Price", min_value=0.0, step=0.01)
        product_stock = st.number_input("Stock", min_value=0, step=1)
        product_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        category_name = st.selectbox("Category", ["Books", "Clothing", "Furniture", "Kitchenware and appliances",
                                                  "Home Decor", "Electronics", "Toys and Games",
                                                  "Sports Equipment", "Bedding and linens", "Personal Care Products"])
        submit = st.form_submit_button("Add Product")

        if submit:
            if product_image is not None:
                # Save the uploaded file to a directory
                image_path = os.path.join("uploads", product_image.name)
                with open(image_path, "wb") as f:
                    f.write(product_image.getbuffer())
            else:
                st.error("Please upload an image.")
                return

            # Get category_id from Categories table based on category_name
            query_category = categories.select().where(categories.c.category_name == category_name)
            result_category = session.execute(query_category).fetchone()
            if result_category:
                category_id = result_category[0]
                new_product = products.insert().values(
                    product_name=product_name,
                    product_condition=product_condition,
                    product_price=product_price,
                    product_stock=product_stock,
                    product_image=image_path,
                    category_id=category_id,
                    seller_id=seller_id
                )
                session.execute(new_product)
                session.commit()
                st.success("Product added successfully!")
            else:
                st.error("Selected category not found in the database.")

    # Display seller's products with checkboxes inside the table and a batch delete button
    st.write("Your Products:")
    query = products.select().where(products.c.seller_id == seller_id)
    result = session.execute(query).fetchall()
    if result:
        df = pd.DataFrame(result, columns=['product_id', 'product_name', 'product_condition', 'product_price', 'product_stock', 'product_image', 'category_id', 'seller_id'])
        
        # Add checkboxes in each row for selection
        selected_product_ids = []
        for index, row in df.iterrows():
            selected = st.checkbox(f"Select {row['product_name']}", key=f"checkbox_{row['product_id']}")
            if selected:
                selected_product_ids.append(row['product_id'])
        
        # Display table excluding product_image, category_id, and seller_id columns for cleaner display
        st.table(df[['product_name', 'product_condition', 'product_price', 'product_stock']])
        
        # Batch delete button at the end of the table
        if st.button("Delete Selected"):
            if selected_product_ids:
                # Delete selected products from database
                delete_query = products.delete().where(products.c.product_id.in_(selected_product_ids))
                session.execute(delete_query)
                session.commit()
                st.success("Selected products deleted successfully.")

    else:
        st.write("No products found.")

#-------LIST IN DATABASE FOR USERS------------
def seller_list():    
    # Function to get sellers from the database
    def get_sellers():
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        select_query = "SELECT * FROM Sellers;"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return rows, columns

    # Function to get products from the database
    def get_products():
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        select_query = "SELECT * FROM Products;"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return rows, columns

    # Streamlit app
    st.title("Sellers and Products")

    # Display Sellers table
    st.header("Sellers")
    st.write("List of Sellers from the database:")
    sellers, seller_columns = get_sellers()
    seller_df = pd.DataFrame(sellers, columns=seller_columns)
    st.table(seller_df)

    # Display Products table
    st.header("Products")
    st.write("List of Products from the database:")
    products, product_columns = get_products()
    product_df = pd.DataFrame(products, columns=product_columns)
    st.table(product_df)
if __name__ == "__main__":
    main()
