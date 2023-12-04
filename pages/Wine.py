import streamlit as st
from utils.WineModel import *
from utils.utils import get_wine_dataset
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

def tab_content(df, Decompositor, decompositor_name):
    st.subheader(f"Dengan {decompositor_name}")
    decompositor = Decompositor(n_components=3)

    df_decompositioned = pd.DataFrame(decompositor.fit_transform(df))
    df_decompositioned['label'] = wine.target.astype(str)
    df_decompositioned['pred_label'] = KMeans(n_clusters=3) \
                                        .fit_predict(df_decompositioned.select_dtypes(float)) \
                                        .astype(str)
    
    original, prediction = st.columns(2)
    with original:
        st.markdown("**Original**")
        st.plotly_chart(
        px.scatter_3d(
            df_decompositioned, 
            0, 1, 2, color='label', 
            height=800, width=800)
        )
    with prediction:
        st.markdown("**Prediksi**")
        st.plotly_chart(
        px.scatter_3d(
            df_decompositioned, 
            0, 1, 2, color='pred_label', 
            height=800, width=800)
        )

    st.markdown(classification_report(df_decompositioned['label'], df_decompositioned['pred_label']))
    
st.header("Algoritma cluster dan klasifikasi Wine menggunakan PCA dan SVD dengan k-Means", divider='rainbow')


st.subheader("Data asli")
wine = get_wine_dataset()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
st.dataframe(df)


st.header("Dekomposisi tanpa melakukan normalisasi", divider='red')

st.subheader('Dekomposisi')
t_pca, t_svd = st.tabs(["PCA", "SVD"])
with t_pca:
    tab_content(df, PCA, "PCA")
with t_svd:
    tab_content(df, TruncatedSVD, "SVD")


st.header("Dekomposisi dengan melakukan normalisasi terlebih dahulu", divider='blue')
for col in df.columns:
    df[col] = MinMaxScaler().fit_transform(df[col].values.reshape(-1, 1))
st.dataframe(df)

st.subheader('Dekomposisi')
t_pca, t_svd = st.tabs(["PCA", "SVD"])
with t_pca:
    tab_content(df, PCA, "PCA")
with t_svd:
    tab_content(df, TruncatedSVD, "SVD")