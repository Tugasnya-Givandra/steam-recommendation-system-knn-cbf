import streamlit as st
from streamlit_option_menu import option_menu

from utils.Model import *
from utils.utils import *
from utils.home_utils import *

if 'user_selection' not in st.session_state:
    st.session_state['user_selection'] = None


st.header("Steam Recommendation System", divider='rainbow')
st.warning("""
        ## Aturan input & panduan sistem
    
        Mohon untuk memasukkan input yang valid, yaitu:
        
        - Minimal 2 judul game yang dimasukkan sebagai input sistem
        - Preferensi penilaian game yang dimasukkan harus memiliki setidaknya 1 rating positif

        Untuk mendapatkan hasil rekomendasi, berikut langkah untuk berinteraksi dengan sistem:
        
        1. Tekan input dropdown dibawah
        2. Ketikkan judul game yang anda ketahui dan atur penilaian dari game yang bersangkutan
        3. Tekan "Get recommendation" untuk mendapatkan hasil rekomendasi
        4. Pindah ke tab "Result" untuk melihat judul game yang direkomendasikan
    """)
st.divider()

st.markdown("<br>", unsafe_allow_html=True)

title_selected = st.multiselect(
    label="Input games you like:",
    options=get_list_of_games(),
    key="user_titles",
    default=st.session_state['user_selection']
)

generate_game_boxes(title_selected)

if st.button('Get recommendation'):
    st.session_state['title_selected'] = title_selected
    with st.spinner("Getting recommendation..."):
        ids = []
        is_recommended = []

        for title in title_selected:
            id = title_to_id[title]

            ids.append(id)
            is_recommended.append(int(st.session_state[id] == "Positive"))
        
        pred_df = pd.DataFrame({
            'app_id': ids,
            'is_recommended': is_recommended
        })

    st.session_state["selected"] = pred_df
    st.success("Go to result page from sidebar to view top 10 recommendations.")

    # nav_to('/Results')