import numpy as np
import shutil as shell
import os
import json

from scipy.stats import linregress
from predict.model.interfaces import IModel
from datetime import datetime

class NoPrediction(IModel):
    """
        Nothing to predict, return average value only
    """
    def __init__(self, name, inputlen=7, outputlen=1):
        """
            @param name: Set model name to identify the model in local dir
            @param inputlen: Set model input length
            @param outputlen: Set model prediction length
        """
        IModel.__init__(self, 'nothing', name)
        self.__InputLength = inputlen
        self.__OutputLength = outputlen

    #--------------------------------- implemented method ---------------------------------
    def fit(self, x, y):
        self.updatecheckpoint()
        pass

    
    def inputrequire(self):
        return self.__InputLength


    def maxpredict(self):
        return self.__OutputLength


    def predict(self, x):
        """
            Predict nothing
        """
        return np.asarray([np.mean(x, axis=0)])
    #--------------------------------- implemented method ---------------------------------
