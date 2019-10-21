from predict.model.interfaces import IModel
from statsmodels.tsa.arima_model import ARIMA as arima
import numpy as np
import warnings

class ARIMA(IModel):

    def __init__(self, name, inputlen=30, outputlen=1):
        IModel.__init__(self, 'arima', name)
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
        x = np.asarray(x)
        pred = []
        for dim in range(x.shape[1]):
            model = arima(x[-self.__InputLength:,dim], (1, 1, 0)).fit(method='mle')
            predict = model.forecast()[0]
            pred.append(predict)
        pred = np.asarray(pred).transpose()

        return pred

if __name__ == "__main__":
    model = ARIMA('test', inputlen=4, outputlen=1)
    #print(model.lastcheckpoint())
    #model.resetmodel()
    #x = np.linspace(0,1,4*10).reshape([-1,4,1])
    #y = np.sum(x, axis=1).reshape([-1,1,1]) / 4
    #model.fit(x, y)
    #model.eval_with_numpy_input(x, y)
    pred = model.predict([[0.44], [0.45], [0.55], [0.55]])
    print(pred)




