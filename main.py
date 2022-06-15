
import numpy as np
import streamlit as st
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("Streamlit example")

st.write("""
# Explore different classifier
""")

dataset_name = st.sidebar.selectbox(
    " Select Dataset", ("Iris", "Breast Cancer", "Wine"))

classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("KNN", "SVM", "Random Forest"))


@st.cache
def get_dataset(name):
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif name == "Wine":
        data = datasets.load_wine()

    return data.data, data.target


X, y = get_dataset(dataset_name)

st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


def add_parameter_ui(cif_name):
    params = dict()

    if cif_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif cif_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif cif_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 0.01, 10.0)
        n_estimators = st.sidebar.slider("n_estimators", 1, 1000)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params


params = add_parameter_ui(classifier_name)


def get_classifier(cif_name, params):
    if cif_name == "KNN":
        cif = KNeighborsClassifier(
            n_neighbors=params["K"]
        )
    elif cif_name == "SVM":
        cif = SVC(
            C=params["C"]
        )
    elif cif_name == "Random Forest":
        cif = RandomForestClassifier(
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            random_state=1234
        )

    return cif


cif = get_classifier(classifier_name, params)
