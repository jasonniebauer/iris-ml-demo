import streamlit as st
import pandas as pd
from prediction import predict


st.title('Classifying Iris Flowers')
st.markdown('Model to expirement with classifying iris flowers into \
            setosa, versicolor, virginica')

st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
    st.text('Sepal characteristics')
    sepal_length = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
    sepal_width = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
with col2:
    st.text('Petal characteristics')
    petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
    petal_width = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

if st.button('Predict type of Iris'):
    result = predict(
        pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        )
    )
    st.text(result[0])