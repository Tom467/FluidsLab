
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def plot(pi_group, pi_group_set, markers, key, inverting):
    legend = []
    plt.figure()

    x = pi_group.values
    x_label = pi_group.formula

    y = pi_group_set.pi_groups[0].values
    y_label = pi_group_set.pi_groups[0].formula

    model = LinearRegression().fit(x.reshape((-1, 1)), y)
    r_sq = model.score(x.reshape((-1, 1)), y)
    y_pred = model.predict(x.reshape((-1, 1)))

    if st.checkbox(key + f' - Coefficient of Determination: {round(r_sq,2)}', value=1) or r_sq > .7:
        if inverting:
            if st.checkbox('invert Y', value=False, key=key+'invert Y'):
                y = 1 / pi_group_set.pi_groups[0].values
                y_label = pi_group.formula_inverse
            if st.checkbox('invert X', value=False, key=key+'invert X'):
                x = 1 / pi_group.values
                x_label = pi_group.formula_inverse

        if len(markers) > 0:
            for marker in markers:
                plt.scatter(x[marker[0]: marker[1]], y[marker[0]: marker[1]], marker=marker[2])
                legend.append(marker[3])
            plt.legend(legend)
        else:
            plt.scatter(x, y)

        plt.plot(x, y_pred)
        legend.append('Linear Fit')
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        st.pyplot(plt)


