import pandas as pd
import streamlit as st
from data_reader import Data
import matplotlib.pyplot as plt
from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis

st.title("Dimensional Analysis")
file = "C:/Users/truma/Downloads/test - Swings.csv"
file = st.file_uploader('csv file', type=['csv'], )

df = pd.read_csv(file)
st.write("Here is the dataset used in this analysis:")
st.write(df)

data = Data(df, pandas=True)
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
