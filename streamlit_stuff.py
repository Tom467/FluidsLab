import pandas as pd
import streamlit as st
from pathlib import Path
from data_reader import Data
import matplotlib.pyplot as plt
from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

st.title("Dimensional Analysis")
if st.button('What is Dimensional Analysis?'):
    a = st.button('Hide', key=1)
    intro_markdown = read_markdown_file("readme.md")
    st.markdown(intro_markdown)
    if a or st.button('Hide', key=2):
        st.markdown('')

instructions = 'Make sure the first row in the .csv file contains header information which should follow the following format: Name-units.'
if st.button('Instructions'):
    st.markdown(instructions)
    if st.button('Hide Instructions'):
        st.markdown('')


file = None
file = st.file_uploader('csv file', type=['csv'])

if file is not None:
    ds = pd.read_csv(file)
    st.write("Here is the dataset used in this analysis:")
    st.write(ds)

    data = Data(ds, pandas=True)
    d = DimensionalAnalysis(data.parameters)
    # figure, axes = d.pi_group_sets[0].plot()

    st.header('Figures')
    plt.close('all')
    for pi_group_set in d.pi_group_sets:
        st.header('Repeating Variables:')
        test = [st.write(repeating.name) for repeating in pi_group_set.repeating_variables]
        for i, pi_group in enumerate(pi_group_set.pi_groups[1:]):
            plt.figure()
            plt.scatter(pi_group.values, pi_group_set.pi_groups[0].values)
            plt.xlabel(pi_group.formula)
            plt.ylabel(pi_group_set.pi_groups[0].formula)
            st.pyplot(plt)
