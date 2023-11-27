import streamlit as st
from streamlit_option_menu import option_menu

from utils.Model import *
from utils.utils import *
from utils.home_utils import *

if 'user_preferences' not in st.session_state:
    st.session_state['user_preferences'] = {}


        # st.write(res)


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

preferences = st.multiselect(
    label="Input games you like:",
    options=get_list_of_games(),
    key="user_titles"
)

user_input = game_selected(preferences)

if st.button("Get recommendation"):
    print(preferences)
    st.session_state["user_preferences"] = user_input

    if 'result' in st.session_state:
        del st.session_state['recommendation_list']
    
    with st.spinner("Getting recommendation..."):
        pref_value = []
        for id in preferences_id:
            pref_value.append(int(st.session_state[id] == "Positive"))
        
        pred_df = pd.DataFrame({
            'user_id': [999999] * len(preferences_id),
            'app_id': preferences_id,
            'is_recommended': pref_value
        })

        items = get_games_dataset()
        model = KnnCBF(items)

        res = model.fit_predict(df_pred=pred_df, k=10)
        st.dataframe(res)
        print(res)

    if type(res) == [ValueError, None]:
        st.error("Recommendation failed. Please select with at least 2 games title.")
    
    st.session_state['result'] = res

    nav_to('/Results')