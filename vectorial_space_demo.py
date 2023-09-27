import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

pca = PCA()
model = SentenceTransformer("all-MiniLM-L6-v2")


def initial_data():
    info_to_add = [
        "hola mundo",
        "pinecone es una empresa de tecnología, que ofrece un servicio de vectorialización de datos",
        "la comida italiana es muy rica",
    ]

    info = pd.DataFrame(columns=["key", "vector", "similarity"])
    for i in info_to_add:
        vector_data = model.encode(i)
        info.loc[len(info.index)] = [i, vector_data, 0]
    return info


def set_plot(highligth_point=None, label=None):
    global pca
    pca = PCA(n_components=2)
    _x = pca.fit_transform(st.session_state.count.vector.tolist())
    plt.scatter(_x[:, 0], _x[:, 1], c="red")

    for i, txt in enumerate(st.session_state.count.key.tolist()):
        plt.annotate(txt[:10], (_x[i, 0], _x[i, 1]))

    if highligth_point is not None:
        highligth_point_vector = pca.transform([highligth_point])[0]
        plt.scatter(highligth_point_vector[0], highligth_point_vector[1], c="blue")
        plt.annotate(label, (highligth_point_vector[0], highligth_point_vector[1]))
    return plt


def add_data_to_info():
    my_key = st.session_state.my_key
    vector_data = model.encode(my_key)
    temp_count = st.session_state.count
    temp_count.loc[len(temp_count.index)] = [my_key, vector_data, 0]
    st.session_state.count = temp_count
    st.session_state.my_key = ""
    set_plot()


def remove_data_from_info():
    remove_key = st.session_state.remove_key
    temp_count = st.session_state.count
    temp_count = temp_count[temp_count["key"] != remove_key]
    st.session_state.count = temp_count
    st.session_state.remove_key = ""


def search():
    vector_search = model.encode(st.session_state.search_box)
    temp_count = st.session_state.count
    temp_count["similarity"] = temp_count["vector"].apply(
        lambda x: float(util.cos_sim(vector_search, x)[0][0])
    )
    temp_count = temp_count.sort_values(by=["similarity"], ascending=False)
    st.session_state.search_result = temp_count[["key", "similarity"]]

    st.session_state.plot = set_plot(vector_search, st.session_state.search_box)


st.title("Vectorial Space Demo")
if "count" not in st.session_state:
    st.session_state.count = initial_data()
if "search_result" not in st.session_state:
    st.session_state.search_result = pd.DataFrame(columns=["key", "similarity"])
if "plot" not in st.session_state:
    st.session_state.plot = set_plot()

st.text_input("Add item", key="my_key", on_change=add_data_to_info)
st.text_input("Item to remove", key="remove_key", on_change=remove_data_from_info)
st.write(st.session_state.count)


st.title("Search in the vectorial space")
st.text_input("Search box", key="search_box", on_change=search)
st.write(st.session_state.search_result)

st.pyplot(st.session_state.plot)
