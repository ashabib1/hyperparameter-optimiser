import os
import string
import torch
import random
import optuna
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.metrics
from skimage.io import imread
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, LabelEncoder, MinMaxScaler
from typing import Tuple, Union
from model_config import RandomForestMP, LassoMP, NeuralMP, ConvolutionalMP
from neuralnetworks import NeuralClassifier, NeuralRegressor, ConvolutionalNeuralClassifier

class HyperParamOpt:

    def __init__(self, model, model_config):
    
        self.model_reference = model
        self.model_config = model_config

    def objective(self, trial: optuna.trial._trial.Trial) -> np.float64:

        """Return accuracy for a trial

        For each trial, get recommended a set of hyperparameters from the
        Suggestion function. Evaluate the trial by first fitting the model,
        then predicting the values of the validation set, and finally finding the
        accuracy by comparing the predicted and correct values.

        Args:
        trial (optuna.trial._trial.Trial): Current trial
        
        Returns:
        np.float64: Accuracy for the trial
        """

        model = self.model_reference()
        params = model.get_params()
        properties = self.model_config.schema()["properties"]
        for key in properties:
            val, min, max, options, kind = self.getdefault(properties, key)
            params[key] = self.suggestion(trial, key, val, kind, min, max, options)
            if key == "n_units_layer1":
                for i in range(params["num_layers"]):
                    params[f"n_units_layer{i+1}"] = self.suggestion(trial, f"n_units_layer{i+1}", val, kind, min, max, options)
        model.set_params(**params)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_val)
        acc = self.metric(self.y_val, predictions)
        return acc
    
    def tune(self, X_train, X_val, X_test, y_train, y_val, y_test, metric: sklearn.metrics, direction: str = "minimize", trials: int = 50, jobs: int = 1) -> optuna.trial._frozen.FrozenTrial:
        
        """Find the best tuning hyperparameters

        Input data and a metric in which to measure accuracy, then iterate
        through a given number of trials. After iterating through all of the
        trials, a best combinaton of hyperparameters, given by the optimal
        value of the performance metric, will be returned.

        Args:
        X_train: Predicting data for the training dataset
        X_test: Predicting data for the testing dataset
        y_train: Target data for the training dataset
        y_test: Target data for the testing dataset
        metrics (sklearn.metrics): Measure of accuracy
        direction (str): Maximize or minimize the metric argument

        Returns:
        _type_: Trial number with the hyperparameters that maximized accuracy
        """

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.metric = X_train, X_val, X_test, y_train, y_val, y_test, metric
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=trials, n_jobs=jobs)
        self.evaluate(study.best_params)
        return study.best_trial
    
    def evaluate(self, best_params):

        model = self.model_reference()
        model.set_params(**best_params)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        acc = self.metric(self.y_test, predictions)
        print("Evaluation Accuracy: ", acc)
        return acc

    def suggestion(self, trial: optuna.trial._trial.Trial, name: str, val: float, kind: str, min: float, max: float, options: list) -> Union[list, int, float]:
        
        """Suggest a tuning value

        Optuna will recommend one value for each hyperparameter to use for
        the next iteration of the trial based on the results of the previous
        trials and the possible values the hyperparameters could take.

        Args:
        trial (optuna.trial._trial.Trial): Current trial
        name (str): Name of hyperparameter
        val (float): Default hyperparameter value
        kind (str): Type of values that the hyperparameter takes
        min (float): Minimum value the hyperparameter can take
        max (float): Maximum value the hyperparameter can take
        options (list): Possible values the hyperparameter can take

        Returns:
        Union[list, int, float]: Returns a suggested value
        """

        if kind == "string" or kind == "boolean":
            return trial.suggest_categorical(name, options)
        elif kind == "integer":
            return trial.suggest_int(name, min, max)
        elif kind == "number":
            return trial.suggest_float(name, min, max)
        else:
            return val

    def getdefault(self, properties: dict, key: str) -> Tuple[float, float, float, list, str]:

        """Retrieve default values

        Iterate through the properties dictionary, and retrieve the default
        values for each variable, the type of variable, and the range of
        possible values that the variable could take.

        Args:
        properties (dict): Dictionary holding all default values
        key (str): Name of variable
        
        Returns:
        Tuple[float, float, float, list, str]: Extracted values from dictionary
        """

        val, min, max, options, kind = None, None, None, None, None
        for paramconfig in properties[key]:
            if paramconfig == "default":
                val = properties[key][paramconfig]
            if paramconfig == "minimum" or paramconfig == "exclusiveMinimum":
                min = properties[key][paramconfig]
            if paramconfig == "maximum" or paramconfig == "exclusiveMaximum":
                max = properties[key][paramconfig]
            if paramconfig == "enum":
                options = properties[key][paramconfig]
            if paramconfig == "type":
                kind = properties[key][paramconfig]
        return val, min, max, options, kind

if __name__ == "__main__":

    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

    SEED = 1234

    np.random.seed(SEED)
    random.seed(SEED)

    def train_val_test_split(X, y, train_size):

        X_train, X_, y_train, y_ = train_test_split(X,y,train_size=TRAIN_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(X_,y_,train_size=0.5)
        return X_train, X_val, X_test, y_train, y_val, y_test

    url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/spiral.csv"
    df = pd.read_csv(url, header=0)
    X = df[["X1", "X2"]].values
    y = df["color"].values
    X_train, X_val, X_test, y_train1, y_val1, y_test1, = train_val_test_split(X=X,y=y,train_size=TRAIN_SIZE)
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_train1)
    classes = list(label_encoder.classes_)
    y_train1 = label_encoder.transform(y_train1)
    y_val1 = label_encoder.transform(y_val1)
    y_test1 = label_encoder.transform(y_test1)
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    y_train = np.zeros((len(y_train1), 3))
    y_val = np.zeros((len(y_val1), 3))
    y_test = np.zeros((len(y_test1),3))
    i, j, k = 0, 0, 0
    for val in y_train1:
        y_train[i][val] = 1
        i = i + 1
    for val in y_val1:
        y_val[j][val] = 1
        j = j + 1
    for val in y_test1:
        y_test[k][val] = 1
        k = k + 1
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    obj = HyperParamOpt(NeuralClassifier, NeuralMP)

    obj.tune(X_train, X_val, X_test, y_train, y_val, y_test, r2_score, "maximize")