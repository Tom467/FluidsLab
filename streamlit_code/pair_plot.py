
import copy
import numpy as np
import pandas as pd
import streamlit as st

import seaborn as sns


def pairplot():
    file = st.sidebar.file_uploader('CSV file', type=['csv'])

    if file is not None:
        ds = pd.read_csv(file)
        st.pyplot(sns.pairplot(ds))
