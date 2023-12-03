import streamlit as st
from utils.utils import *
from sklearn.decomposition import TruncatedSVD
import plotly.express as px

st.header("Jeroan jeroan~", divider='rainbow')

df_show = get_games_dataset().copy()

df = pd.read_pickle("datasets/clean_games.pkl")
df_original = df.iloc[:, 13:]

selection = df_original.columns

select_item = st.multiselect("Tags", selection)
col_selection_name = ', '.join(select_item)

svd = TruncatedSVD(n_components=2, random_state=1)
res = svd.fit_transform(df_original)
df_decomposed = pd.DataFrame(res)
df_decomposed['title'] = df['title']
df_decomposed[col_selection_name] = df[select_item].all(axis=1)


st.plotly_chart(
    px.scatter(df_decomposed, 
               0, 1, 
               hover_data=[0, 1, 'title'], color=col_selection_name, 
               opacity=0.5, color_discrete_sequence=px.colors.qualitative.G10,
               width=800, height=800)
)