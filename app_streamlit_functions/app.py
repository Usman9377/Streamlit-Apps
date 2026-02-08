import streamlit as st
import numpy as np
import pandas as pd

# Adding title of your app
st.title("My First Testing App")
# Adding Simple Text
st.write("Muhammad Usman (Scientific Officer)")

# user input
number = st.slider("pick a number", 0, 100)

# print the text of number
st.write(f"you selected: {number} Muhammad Usman")

# adding a button
if st.button("Say Assalam o Aalaikum"):
    st.write("Wa alaikum asaalam, who are you ?")
else:
    st.write("Your friend bro")

# add radio button with options
Usman = st.radio(
    "what's your favorite fruit Usman ?",
    ('Mango', 'Apple', 'Banana'))

# print the text of Usman
st.write(f"You selected: {Usman} Muhammad Usman")

# add a drop down list
# option = st.selectbox(
#     "How would you like to be connected",
#     ("Email", "Home phone", "Mobile phone"))

# add a drop down list on left sidebar
option = st.sidebar.selectbox(
    "How would you like to be connected",
    ("Email", "Home phone", "Mobile phone"))

# add your whatsapp number
st.sidebar.text_input("Enter your whatsapp number")

# add a file uploader
uploaded_file = st.sidebar.file_uploader("Choos a CSV file", type = "csv")

# Create a line chart
data = pd.DataFrame({
    "first column": list(range(1, 11)),
    "second column": np.arange(number, number + 10)
})

st.line_chart(data)