import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from PIL import Image

#The tittle, emoji from webfx emoji cheat sheet
st.set_page_config(page_title="My Preloved item Recommendation", page_icon="😁", layout= "wide")

#---------Side navigation--------------
with st.sidebar:
  selected = option_menu(
    menu_title = "Menu",
    options=["Home", "Register", "Login"],
    icons=["house", "pencil-square", "arrow-down-right-square-fill"],
    menu_icon="cast",
    default_index=0,
  )

#---------selecting side bar------------
if selected == "Home":
   st.title(f"You are at Homepage {selected}")
if selected == "Register":
   st.title(f"You are at Register page {selected}")
if selected == "Login":
   st.title(f"You are at Login page {selected}")


#--------to Set up the animation--------
def load_lottieurl(url):
  r = requests.get(url)
  if r.status_code != 200:
    return None
  return r.json()
lottie_coding = load_lottieurl("https://lottie.host/37713d2b-ffd6-48db-b024-ffc348d32248/C1J6R7FTKt.json"
)
img_contact_form = Image.open("images/shopping.png")

#-----HEADER SECTION --------
with st.container():
  st.subheader("Hi, Welcome to my recommender website")
  st.title("What is this website for?")
  st.write("This website will help you find the product you need and the seller to contact with")

#------ MAIN PAGE ------------
with st.container():
  st.write("----")
  left_column, right_column = st.columns(2)
  with left_column:
    st.header("Register")
    #st.write("##")
    st.write(""" 
    So here you will register as a customer, you will insert you name, age, gender for the demographic and product category, condition and price for your 
    product preference.
    """) 

    st.write("[Youtube Channel >](https://www.youtube.com/watch?v=Pe4gUSIxte8)")

#--------animation on the right side------
  with right_column:
    st_lottie(lottie_coding, height=300, key="coding")

with st.container():
  st.write("----")
  st.header("My projects")
  image_column, text_column = st.columns((1,2))
  with image_column:
      st.image(img_contact_form)
  with text_column:
      st.subheader("Integrate Lottie Animations Inside Your Streamlit App")
      st.write(
        """
        Learn How to use Lottie FIles in Streamlit!
        """
      )
    
with st.container():
  image_column, text_column = st.columns((1,2))
  with image_column:
      st.image(img_contact_form)
  with text_column:
      st.subheader("Integrate Lottie Animations Inside Your Streamlit App")
      st.write(
        """
        Learn How to use Lottie FIles in Streamlit!
        """
      )

#-------Contacts--------
with st.container():
   st.header("Contacts")
   st.write("Email: ")
   st.markdown("[Email](https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox)")
