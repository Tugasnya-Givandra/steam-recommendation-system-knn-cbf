import pandas as pd
import streamlit as st
import requests

from io import BytesIO
from PIL import Image

@st.cache_data
def cache_request(url):
    return requests.get(url).content

@st.cache_data
def load_id_to_title():
    return pd.read_pickle("datasets/id_to_title.pkl")

@st.cache_data
def load_title_to_id():
    return pd.read_pickle("datasets/title_to_id.pkl")
    
@st.cache_data
def get_list_of_games():
    return list(pd.read_pickle("datasets/title_to_id.pkl"))

@st.cache_data
def get_games_dataset():
    dataset = pd.read_pickle("datasets/clean_games.pkl")
    return dataset

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)


id_to_title = load_id_to_title()
title_to_id = load_title_to_id()