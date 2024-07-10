import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from PIL import Image
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker

# Database connection setup
DATABASE_URI = 'mysql+pymysql://root:@localhost:3307/prelovedrecommender'
engine = create_engine(DATABASE_URI)
metadata = MetaData()
metadata.bind = engine
Session = sessionmaker(bind=engine)
session = Session()

# The title and emoji from webfx emoji cheat sheet
st.set_page_config(page_title="My Preloved Item Recommendation", page_icon="😁", layout="wide")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Register", "Login"],
        icons=["house", "pencil-square", "arrow-down-right-square-fill"],
        menu_icon="cast",
        default_index=0,
    )

# Main function to display pages
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if selected == "Home":
        st.title(f"You are at Homepage {selected}")
    elif selected == "Register":
        register(session)
    elif selected == "Login":
        login(session)

    # Header section
    with st.container():
        st.subheader("Hi, Welcome to my recommender website")
        st.title("What is this website for?")
        st.write("This website will help you find the product you need and the seller to contact with")

    # Main page
    with st.container():
        st.write("----")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("Register")
            st.write(""" 
            So here you will register as a customer, you will insert your name, age, gender for the demographic and product category, condition and price for your 
            product preference.
            """)
            st.write("[YouTube Channel >](https://www.youtube.com/watch?v=Pe4gUSIxte8)")
        with right_column:
            lottie_coding = load_lottieurl("https://lottie.host/37713d2b-ffd6-48db-b024-ffc348d32248/C1J6R7FTKt.json")
            st_lottie(lottie_coding, height=300, key="coding")

    with st.container():
        st.write("----")
        st.header("My projects")
        image_column, text_column = st.columns((1, 2))
        with image_column:
            img_contact_form = Image.open("images/shopping.png")
            st.image(img_contact_form)
        with text_column:
            st.subheader("Integrate Lottie Animations Inside Your Streamlit App")
            st.write(
                """
                Learn How to use Lottie Files in Streamlit!
                """
            )

    with st.container():
        image_column, text_column = st.columns((1, 2))
        with image_column:
            img_contact_form = Image.open("images/shopping.png")
            st.image(img_contact_form)
        with text_column:
            st.subheader("Integrate Lottie Animations Inside Your Streamlit App")
            st.write(
                """
                Learn How to use Lottie Files in Streamlit!
                """
            )

    # Contacts section
    with st.container():
        st.header("Contacts")
        st.write("Email: ")
        st.markdown("[Email](https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox)")

# Load Lottie animation
def load_lottieurl(url):
    import requests
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if __name__ == "__main__":
    main()