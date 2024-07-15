
import seaborn as sns
import streamlit as st

from streamlit_code.streamlit_util import read_markdown_file
from general_dimensional_analysis.data_reader import Data


def pairplot(group):
    # with st.expander('Additional Information on Pairplot'):
    #     st.markdown(read_markdown_file(''))
    df = Data.group_to_dataframe_without_units(group)
    options = list(df.columns)
    options.insert(0, None)
    selected = st.sidebar.selectbox('Variable in data to map plot aspects to different colors', options)
    st.pyplot(sns.pairplot(df, hue=selected))
