import streamlit as st
from utils.utils import *

st.header("Jeroan jeroan~", divider='rainbow')

df_show = get_games_dataset().copy()
st.dataframe()