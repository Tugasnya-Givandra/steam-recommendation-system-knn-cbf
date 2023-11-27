import streamlit as st

st.header("About", divider='rainbow')

st.subheader("Dataset")
"""
Untuk proyek ini, saya menggunakan data 
[*Game Recommendations on Steam*](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam), 
yang disediakan oleh Anton Kozyriev. Data yang digunakan dalam penelitian ini adalah informasi data per tanggal 27 Nov 2023. 
Berisi informasi pengguna yang dianonimkan, informasi game, serta interaksi antara pengguna dan item.
"""

st.subheader("Algoritma")
"""
KNN CBF

-- PLACEHOLDER --
"""


st.markdown(
        """
        > This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). 
        > Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. 
        > This means it can work with sparse matrices efficiently.
        """
    )