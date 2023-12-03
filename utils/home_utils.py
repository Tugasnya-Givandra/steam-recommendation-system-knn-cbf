import streamlit as st
from utils.utils import *
from utils.Model import KnnCBF

def generate_game_boxes(titles):
    for title in titles:
        app_id = title_to_id[title]
        url = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/header.jpg"

        with st.container():
            img_col, pref_col = st.columns([3, 2])

            img_col.image(BytesIO(cache_request(url)))
            pref_col.selectbox(
                "Your rating:",
                options=["Positive", "Negative"],
                key=app_id,
            )
    