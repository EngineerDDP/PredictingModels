import numpy as np  
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

import shutil as shell

from predict.model.interfaces import IModel
from datetime import datetime

from threading import Lock


class MLPPredictLocal(IModel):
    """
        Multi-layer perceptron
    """

    _Lock = Lock()

    def __init__(self, name, inputlen=30, outputlen=21, activation=tf.nn.tanh):
        """
            @param name: Set model name to identify the model in local dir
            @param inputlen: Set model input length
            @param outputlen: Set model prediction length
            @param path: Set path to save the model

            Config most parameters in here, instead of the __model_fn()
        """
        IModel.__init__(self, 'mlp', name)
        self.__InputLength = inputlen
        self.__OutputLength = outputlen
        self.__Act = activation
        self.__InitScale = 0.1
        self.__BatchSize = 32
        self.__HiddenLayerUnits = [32,32,32,32]
        self.__TrainEpoches = 2200

        self.nn = tf.estimator.Estimator(model_fn=self.__model_fn, model_dir=self.ModelPath)


    #--------------------------------- implemented method ---------------------------------
    def fit(self, x, y):
        # save dimension parameters
        self.Config['Dims'] = x.shape[2]
        # reshape inputs
        assert x.shape[2] == y.shape[2], 'Shapes not match between inputs and outputs, mlp.py line 41.'
        x = x.reshape([x.shape[0], x.shape[1] * x.shape[2]])
        y = y.reshape([y.shape[0], y.shape[1] * y.shape[2]])
        # save normalization parameters
        self.Config['Std'] = x.std()
        self.Config['Mean'] = x.mean()
        # normalize
        x = (x - self.Config['Mean']) / self.Config['Std']
        y = (y - self.Config['Mean']) / self.Config['Std']

        MLPPredictLocal._Lock.acquire()
        self.train_with_numpy_input(x, y, False)
        MLPPredictLocal._Lock.release()

        self.updatecheckpoint()
        

    def inputrequire(self):
        return self.__InputLength


    def maxpredict(self):
        return self.__OutputLength


    def predict(self, x):
        x = np.asarray(x)
        # reshape x as one sample row
        x = x.reshape([1, self.__InputLength * self.Config['Dims']])
        # normalize inputs
        x = (x - self.Config['Mean']) / self.Config['Std']

        # get result
        result = list(self.predict_with_numpy_input(x))

        # denormalize outputs
        result = result[0]['predictions'] * self.Config['Std'] + self.Config['Mean']
        # reshape as origin
        result = result.reshape([self.__OutputLength, self.Config['Dims']])

        return result


    def resetmodel(self):
        self.nn = None
        shell.rmtree(self.ModelPath, ignore_errors=True)
        self.nn = tf.estimator.Estimator(model_fn=self.__model_fn, model_dir=self.ModelPath)
    #--------------------------------- implemented method ---------------------------------
        

    def __xavier_init(self, shape):
        """
            Initialize weight depends on units counts
            to make sure that input for next layer units
            no bigger that 1 ( to keep good gradient descent
            speed )
        """
        low = 0
        high = self.__InitScale * np.sqrt(1.0/(shape[0]))
        return tf.random_uniform(shape, minval = low, maxval = high, dtype = tf.float32)


    def __init_weight(self, shape, name='weight'):
        """
            Initialize weight with given shape
        """
        inita_w = self.__xavier_init(shape)
        w = tf.Variable(inita_w,dtype=tf.float32,name=name)
        return w


    def __init_bias(self, shape, name='bias'):
        """
            Initialize bias with given shape
        """
        inita_b = tf.constant(0.1,shape=shape)
        b = tf.Variable(inita_b,dtype=tf.float32,name=name)
        return b


    def __nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act):
        """
            Initialize fully connected layer
        """
        with tf.name_scope(layer_name):

            with tf.name_scope('weights'):
                weight = self.__init_weight([input_dim,output_dim])

            with tf.name_scope('bias'):
                bias = self.__init_bias([output_dim])

            with tf.name_scope('linear_compute'):
                preact = tf.matmul(input_tensor,weight)+bias

            if act is not None:
                activeation = act(preact,name = 'activation')
            else:
                activeation = preact

            return activeation


    def __initialize(self, x, y, mode):
        """
            Build up neural network
        """

        INPUT_LENGTH = self.__InputLength * self.Config['Dims']
        OUTPUT_LENGTH = self.__OutputLength * self.Config['Dims']
        HIDDEN_LAYER = self.__HiddenLayerUnits

        hidden_layer = self.__nn_layer(x, input_dim=INPUT_LENGTH, output_dim=HIDDEN_LAYER[0], layer_name='hidden_layer1', act=self.__Act)

        for i in range(len(HIDDEN_LAYER) - 1):
            layer_name='hidden_layer{}'.format(i+2)
            hidden_layer = self.__nn_layer(hidden_layer, HIDDEN_LAYER[i], HIDDEN_LAYER[i+1], layer_name, self.__Act)

        # use dropout layer dropout_prob for keep probabilities 1.0 for keep all 0.0 for keep none
        with tf.name_scope('dropout'):
            dropout = tf.layers.dropout(hidden_layer, training=mode == tf.estimator.ModeKeys.TRAIN)

        # output layer
        self.prediction = self.__nn_layer(dropout, HIDDEN_LAYER[-1], OUTPUT_LENGTH, layer_name='out_layer', act=None)

        if mode != tf.estimator.ModeKeys.PREDICT:
            # MSE
            with tf.name_scope('loss'):
                self.mse = tf.losses.mean_squared_error(labels=y, predictions=self.prediction)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss=self.mse, global_step=tf.train.get_global_step())

        return;
    

    def __model_fn(self, features, labels, mode):
        """
            Model function, for creating estimators to
            train, evaluate or predict using the model
        """
        features = tf.cast(features['x'], tf.float32)
        if labels is not None:
            labels = tf.cast(labels, tf.float32)

        self.__initialize(features, labels, mode)

        attributes = {
            "predictions":      self.prediction
            }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,predictions=attributes)

        eval_metric_ops = {
            "mean_square_error":    tf.metrics.mean_squared_error(labels, self.prediction),
            "relative_error":       tf.metrics.mean_relative_error(labels, self.prediction, normalizer=tf.ones_like(labels))
            }
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode,loss=self.mse,eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode,loss=self.mse,train_op=self.train_op)
        else:
            return None;


    def train_with_numpy_input(self, x ,y, debug=True):
        """
            Train model with numpy input
        """
        input_func = tf.estimator.inputs.numpy_input_fn(
            x={'x':x},
            y=y,
            batch_size = self.__BatchSize,
            num_epochs = None,
            shuffle = True);
        if debug:
            tensors_to_log = {"loss":"loss/mean_squared_error/value:0"}
            hook = tf.train.LoggingTensorHook(tensors_to_log,every_n_iter=50)
            self.nn.train(input_fn=input_func, steps=self.__TrainEpoches, hooks=[hook])
        else:
            self.nn.train(input_fn=input_func, steps=self.__TrainEpoches)

        return;


    def eval_with_numpy_input(self, x, y):
        input_func = tf.estimator.inputs.numpy_input_fn(
            x={'x':x},
            y=y,
            num_epochs=1,
            shuffle=False);
        result = self.nn.evaluate(input_fn=input_func)
        return result;


    def predict_with_numpy_input(self, x):
        input_func = tf.estimator.inputs.numpy_input_fn(
            x={'x':x},
            y=None,
            num_epochs=1,
            shuffle=False);
        result = self.nn.predict(input_fn=input_func)
        return result;


if __name__ == '__main__':
    model = MLPPredictLocal('test', inputlen=4, outputlen=1)
    print(model.lastcheckpoint())
    #model.resetmodel()

    x = np.linspace(0,1,4*10).reshape([-1,4,1])
    y = np.sum(x, axis=1).reshape([-1,1,1]) / 4

    model.fit(x, y)
    #model.eval_with_numpy_input(x, y)
    pred = model.predict([[0.44], [0.45], [0.55], [0.55]])
    print(pred)