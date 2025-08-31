import streamlit as st
import pandas as pd
st.title("Streamli Text Input")
name=st.text_input("Enter your name:")


age=st.slider("select your age:",0,100,25)

options=["python","java","c++","javascript"]
choice=st.selectbox("choose your favorite language:",options)
st.write(f"you selected {choice}.")
st.write(f"your age is {age}")
if name:
    st.write(f"hello,{name}")
    
data={
    "name":["john","kirti","tuktuk","jill"],
    "Age":[23,45,67,56],
    "City":["kanpur","varanasi","meerut","goa"]
    
}
df=pd.DataFrame(data)
df.to_csv("sampledata.csv")
st.write(df)

uploaded_file=st.file_uploader("choose a csv file",type="csv")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)