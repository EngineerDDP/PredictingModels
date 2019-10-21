import numpy as np

from predict.model.interfaces import IModel


class PolyFit(IModel):

    def __init__(self, name, inputlen=7, outputlen=1):
        IModel.__init__(self, 'poly', name)
        self.__InputLength = inputlen
        self.__OutputLength = outputlen


    def fit(self, x, y):
        self.updatecheckpoint()
        pass


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
            f = np.polyfit(t, x[-self.__InputLength:,dim], 3)
            poly3 = np.poly1d(f)

            pred.append(poly3(t_pred))

        pred = np.asarray(pred).transpose()

        return pred
