# PredictingModels
Models used for sequential data prediction.
# Structure
## Folder:model
Raw models saved in here. All models in model folder have implemented interface *model.interfaces.IModel*
### arima.py
ARIMA (Autoregressive Integrated Moving Average model) implemented in this module. <br>
ARIMA class was an adaptor of *statsmodels.tsa.arima_model.ARIMA*.
### lsr.py
LSR (Least Square Estimation) implemented in this module. <br>
LSR class was an adaptor of function linregress in *scipy.stats*.
### mlp.py
MLP (Multi-Layer Perceptron) implemented in this module. <br>
MLP implemented using APIs in tensorflow.
MLP contains a customized *Estimator*.
### mlpLocal.py
MLP (Multi-Layer Perceptron) implemented in this module. <br>
MLP implemented using APIs in numpy.
### poly.py
Polynomial fitting implemented in this module. <br>
Polynomial fitting implemented using APIs in numpy.
### nopredict.py
Nothing to do in here, implemented for project structural integrity.
## exceptions.py
Contains exceptions that may occurred within this module.
## factory.py
Model factory class here. <br>
Class used for building algorithm modules.
## interfaces.py
Defined the interface of global prediction module.
## predicting.py
Global interface, can build models using string parameters, and do predicting using unified function calls.
# Requirements
* Python 3.6
* numpy <any version>
* python datetime module
* scipy <any version>
* statsmodels <any version>
* tensorflow <required if you want to use the tensorflow version of mlp>
