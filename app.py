"""
Sentiment Analysis Application
Author: Tanay Tunçer
"""

import streamlit as st
import pickle
from logistic_regression import LogisticRegression


st.title("Sentiment Analysis App")

st.text("Lorem ipsum")
st.text_input("")

st.button("Submit")

st.markdown("---")
st.markdown(
    "More information about the machine learning model can be found [here](https://github.com/tanaytuncer/sentiment_analysis)")