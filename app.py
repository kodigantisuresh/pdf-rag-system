import streamlit as st
from config import APP_NAME, APP_VERSION

st.set_page_config(
    page_title=APP_NAME,
    layout="wide"
)

st.title(APP_NAME)
st.write(f"Version : {APP_VERSION}")
st.success("Project Setup Completed Successfully!")

