import streamlit as st

#The tittle, emoji from webfx emoji cheat sheet
st.set_page_config(page_title="My Preloved item Recommendation", page_icon="😁", layout= "wide")

#-----HEADER SECTION --------
with st.container()
  st.subheader("Hi, Welcome to my recommender website")
  st.title("What is this website for?")
  st.write("This website will help you find the product you need and the seller to contact with")

#------ MAIN PAGE ------------
with st.container():
  st.write("----")
  left_column, right_column = st.columns(2)
  with left_column:
    st.header("Register")
    st.write("##")
    st.write(""" 
    So here you will register as a customer, you will insert you name, age, gender for the demographic and product category,condition and price for your 
    product preference.
    """) 

    st.write("[Youtube Channel >](https://www.youtube.com/watch?v=Pe4gUSIxte8)")
