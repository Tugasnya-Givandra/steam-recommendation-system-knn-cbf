import streamlit as st
import requests
from utils.Model import KnnCBF
from utils.utils import *

from io import BytesIO
from PIL import Image

def generate_res_gameboxes(app_ids):
    for app_id in app_ids:
        url_page = f"https://store.steampowered.com/app/{app_id}/"
        url_img = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/header.jpg"
        resp = requests.get(url_img)

        if resp.status_code == 200:
            with st.container():
                st.image(BytesIO(resp.content))
                # st.caption(data_ids[id])
                st.caption(f"[{id_to_title[app_id]}]({url_page})")
            
        st.divider()

if 'selected' not in st.session_state:
    st.error('Please input preferences titles and run "Get recommendation"')
else:
    df_pred = st.session_state['selected']

    items = get_games_dataset()
    model = KnnCBF(items)
    df_res = model.fit_predict(df_pred, k=10)

    generate_res_gameboxes(df_res['app_id'])

    # st.dataframe(df_res)

    # res = model.fit_predict(df_pred=pred_df, k=10)
    # if type(res) == [ValueError, None]:
    # st.error("Gagal membuat rekomendasi...")

    #  st.success(f"Top {len(st.session_state['input'])}")
    # generate_res_gameboxes(st.session_state['rs'])
