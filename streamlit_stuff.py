import pandas as pd
import streamlit as st
from pathlib import Path
from data_reader import Data
import matplotlib.pyplot as plt
from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


st.set_page_config(layout="wide")
st.title("Dimensional Analysis")
with st.expander('What is Dimensional Analysis?'):
    intro_markdown = read_markdown_file("readme.md")
    st.markdown(intro_markdown)
    # if a or st.button('Hide', key=2):
    #     st.markdown('')
# with st.expander('About this app'):
instructions = 'Make sure the first row in the .csv file contains header information which should follow the following format: Name-units.'
with st.expander('Instructions'):
    st.markdown(instructions)


file = None
st.sidebar.header('CSV File')
file = st.sidebar.file_uploader('csv file', type=['csv'])

if file is not None:
    ds = pd.read_csv(file)
    st.sidebar.write("Here is the dataset used in this analysis:")
    st.sidebar.write(ds)

    data = Data(ds, pandas=True)
    d = DimensionalAnalysis(data.parameters)
    # figure, axes = d.pi_group_sets[0].plot()

    st.subheader('Generating Possible Figures')
    plt.close('all')
    my_bar = st.progress(0)
    for h, pi_group_set in enumerate(d.pi_group_sets):
        st.subheader('Repeating Variables:')
        text = pi_group_set.repeating_variables[0].name
        for repeating in pi_group_set.repeating_variables[1:]:
            text += ', ' + repeating.name
        st.write(text)
        # test = [st.write(repeating.name) for repeating in pi_group_set.repeating_variables]
        for i, pi_group in enumerate(pi_group_set.pi_groups[1:]):
            plt.figure()
            plt.scatter(pi_group.values, pi_group_set.pi_groups[0].values)
            plt.xlabel(pi_group.formula, fontsize=14)
            plt.ylabel(pi_group_set.pi_groups[0].formula, fontsize=14)
            st.pyplot(plt)
        my_bar.progress((h+1) / len(d.pi_group_sets))
    st.header('Figures')
    st.balloons()
