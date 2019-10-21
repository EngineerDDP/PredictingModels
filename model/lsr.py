import numpy as np
import shutil as shell
import os
import json

from scipy.stats import linregress
from predict.model.interfaces import IModel
from datetime import datetime


class LeastSquareEstimator(IModel):
    """
        Ordinary Linear Least Square Regression
    """
    def __init__(self, name, inputlen=7, outputlen=1):
        """
            @param name: Set model name to identify the model in local dir
            @param inputlen: Set model input length
            @param outputlen: Set model prediction length
            @param path: Set path to save the model

            Config most parameters in here, instead of the __model_fn()
        """
        IModel.__init__(self, 'ols', name)
        self.__InputLength = inputlen
        self.__OutputLength = outputlen

    #--------------------------------- implemented method ---------------------------------
    def fit(self, x, y):
        self.updatecheckpoint()

    
    def inputrequire(self):
        return self.__InputLength


    def maxpredict(self):
        return self.__OutputLength


    def predict(self, x):
        """
            Predict each dimension separately
        """
        x = np.array(x)
        t = np.arange(0, self.__InputLength, 1)
        t_pred = np.arange(self.__InputLength, self.__InputLength + self.__OutputLength, 1)

        pred = []

        for dim in range(x.shape[1]):

            a, b, r, p, std = linregress(t, x[-self.__InputLength:,dim])
            pred.append(a * t_pred + b)

        pred = np.asarray(pred).transpose()

        return pred
    #--------------------------------- implemented method ---------------------------------


if __name__ == "__main__":
    model = LeastSquareEstimator('name', inputlen=4, outputlen=2)

    print(model.lastcheckpoint())

    x = np.linspace(0,1,4*10).reshape([-1,4])
    y = np.sum(x, axis=1).reshape([-1,1])

    model.fit(x, y)
    #model.eval_with_numpy_input(x, y)
    pred = model.predict([[0.44,0.1], [0.45,0.2], [0.55,0.3], [0.55,0.4]])
    print(pred)