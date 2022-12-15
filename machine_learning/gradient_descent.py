import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class GradientDescent:
    def __init__(self, x, y, degree=3):
        self.x_train, self.y_train, self.x_cross, self.y_cross, self.x_test, self.y_test = self.data_divider(x, y)
        self.coefficients = None
        self.model = Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression(fit_intercept=False))])
        self.fit_model()

    def fit_model(self):
        self.model = self.model.fit(self.x_train, self.y_train)
        print('shape', self.x_train.shape)
        self.coefficients = np.round(self.model.named_steps['linear'].coef_, 2)
        print(self.coefficients)

    def predict(self, x_values):
        prediction = self.model.predict(x_values)
        return prediction

    def plot(self):
        figure = plt.figure(2)
        axis = figure.add_subplot(111)
        y1 = self.predict(self.x_train)
        axis.plot(y1, y1, c='r', label='predicted')
        axis.scatter(y1, self.y_train, s=10, c='b', marker="s", label='measured')
        plt.legend(loc='upper left')

    def error(self, x, y):
        y1 = self.predict(x)
        error = sum((y-y1)**2)/(y.shape[0]-1)
        return error

    @staticmethod
    def data_divider(x, y):
        train = []
        cross = []
        test = []
        for i in range(y.shape[0]):
            if i % 8 == 0:
                cross.append(i)
            elif (i - 1) % 8 == 0:
                test.append(i)
            else:
                train.append(i)
        return (np.array([x[i] for i in train]),
                np.array([y[i] for i in train]),
                np.array([x[i] for i in cross]),
                np.array([y[i] for i in cross]),
                np.array([x[i] for i in test]),
                np.array([y[i] for i in test]))


if __name__ == "__main__":
    x0 = np.random.rand(100, 1)
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    x3 = np.random.rand(100)
    x_data = np.array([x1, x2, x3]).T
    y0 = 4 * x0**2
    y_data = 4 * x_data[:, 0]**2 + 2 * x_data[:, 0]**2 * x_data[:, 1]**2 + x_data[:, 2]**3
    regression = GradientDescent(x0, y0)

    # test1 = regression.predict(x_data)
    # print(test1)

    error1 = regression.error(x0, y0)
    print('error', error1)


