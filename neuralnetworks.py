import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch import Tensor, nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from sklearn.metrics import r2_score
from pydantic import BaseModel, confloat, conint
from typing import Literal, Tuple

class NeuralClassifier(nn.Module):

    def __init__(self):

        super(NeuralClassifier, self).__init__()
        self.params_dict = self.get_params()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:

        """Forward Propagation
        This function does the forward propagaion for the neural network, meaning that the
        input data is fed into the forward direction through the network. Each hidden layer
        takes its input data, processes it as per the activation functoin and passes it to
        the next layer.

        Args:
        x_in (torch.Tensor): Input data of the neural network

        Returns:
        torch.Tensor: Forward propagation, returning the output
        """

        for layer in self.hidden:
            x_in = self.params_dict['activation'](layer(x_in))
            x_in = nn.Dropout(self.params_dict['dropout_p'])(x_in)
        x_out = F.softmax(self.out(x_in), dim=1)
        return x_out
    
    def layers(self, input_dim: int, hidden_dims: list, num_classes: int) -> Tuple[list,torch.nn.modules.linear.Linear]:

        """Assemble layers together

        This function applies a linear transformation for each step of the neural network
        model, and then returns said layers into two variables: a list of all the layer
        transformations, and then the final layer transformation.

        Args:
        input_dim (int): Dimension of the input
        hidden_dims (list): Dimension of each hidden dimension
        num_classes (int): Dimensoin of the output

        Returns:
        Tuple[list, torch.nn.modules.linear.Linear]: Return the built layers
        """

        if hidden_dims == []:
            return [], nn.Linear(input_dim, num_classes)
        hide = []
        hide.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            hide.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        final = nn.Linear(hidden_dims[-1], num_classes)
        return hide, final
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):

        """Fits the training data on to the target

        This function adjusts the weights according to the data values, so that
        a higher value of accuracy can be achieved. This done for a set number
        of epochs. Nothing is returned.

        Args:
        X (torch.Tensor): The training X data
        y (torch.Tensor): The training y data
        """

        self.in_dim, self.out_dim = X.shape[1], y.shape[1]
        self.hidden, self.out = self.layers(self.in_dim, self.params_dict['hidden_units'], self.out_dim)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=self.params_dict['learning_rate'])
        for epoch in range(self.params_dict['epochs']):
            y_pred = self.forward(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                accuracy = self.accuracy_fn(y_pred=y_pred, y_true=y)
                print(f"Epoch: {epoch} | loss: {loss:.2f}, accuracy: {accuracy:.1f}")

    def predict(self, X: torch.Tensor) -> np.ndarray:

        """Predicts the output value

        This function intakes the testing set of data, and then predicts the values
        of the output based on the prevoiusly fitted model on the training set.

        Args:
        X (torch.Tensor): The testing X data

        Returns:
        np.ndarray: The prediction of the testing data
        """

        predictions = self.forward(X)
        predictions = predictions.argmax(axis=1)
        y_predictions = np.zeros((len(X), self.out_dim))
        i = 0
        for val in predictions:
            y_predictions[i][val] = 1
            i = i + 1
        y_predictions = torch.Tensor(y_predictions)
        return y_predictions
    
    def get_params(self) -> dict:

        """Returns default hyperparameter values

        This function just returns the names of of the hyperparameters that we will
        be analyzing, and their default values.

        Returns:
        dict: Default values of the hyperparameters
        """

        dict = {'activation': F.relu, 'hidden_units': [], 'dropout_p': 0.0, 'learning_rate': 1e-2, 'epochs': 500, 'batch_size': 32}
        return dict
    
    def set_params(self, **params: dict) -> dict:

        """Updates the dictionary parameters

        This function takes the hyperparameter values that Optuna recommended for
        the next trial, and sets these values into the dictionary.

        Returns:
        dict: Optuna picked hyperparameters values
        """

        for key in params:
            if params[key] == "F.relu":
                self.params_dict[key] = F.relu
            elif params[key] == "F.sigmoid":
                self.params_dict[key] = F.sigmoid
            elif params[key] == "F.tanh":
                self.params_dict[key] = F.tanh
            else:
                self.params_dict[key] = params[key]
        for i in range(self.params_dict["num_layers"]):
            self.params_dict['hidden_units'].append(params[f'n_units_layer{i+1}'])
        return self.params_dict
    
    def accuracy_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:

        """Finds the accuracy

        This function checks how many data points were classified correctly,
        and how many were classified incorrectly. Then, the function returns
        a values from zero to one depending on how well the data points were
        classified.

        Args:
        y_pred (torch.Tensor): Our predictions for the y test values
        y_true (torch.Tensor): The correct y test values

        Returns:
        float: Returns accuracy
        """

        n_correct = torch.eq(y_pred.argmax(axis=1), y_true.argmax(axis=1)).sum().item()
        accuracy = (n_correct / len(y_pred)) * 100
        return accuracy
    
class NeuralRegressor(nn.Module):

    def __init__(self):

        super(NeuralRegressor, self).__init__()
        self.params_dict = self.get_params()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:

        """Forward Propagation

        This function does the forward propagaion for the neural network, meaning that the
        input data is fed into the forward direction through the network. Each hidden layer
        takes its input data, processes it as per the activation functoin and passes it to
        the next layer.

        Args:
        x_in (torch.Tensor): Input of the neural network

        Returns:
        torch.Tensor: Forward propagation
        """

        for layer in self.hidden:
            x_in = self.params_dict['activation'](layer(x_in))
            x_in = nn.Dropout(self.params_dict['dropout_p'])(x_in)
        x_out = F.relu(self.out(x_in))
        return x_out
    
    def layers(self, input_dim: int, hidden_dims: list, num_classes: int) -> Tuple[list, torch.nn.modules.linear.Linear]:

        """Assemble layers together

        This function applies a linear transformation for each step of the neural network
        model, and then returns said layers into two variables: a list of all the layer
        transformations, and then the final layer transformation.

        Args:
        input_dim (int): _description_
        hidden_dims (list): _description_
        num_classes (int): _description_

        Returns:
        Tuple[list, torch.nn.modules.linear.Linear]: _description_
        """

        if hidden_dims == []:
            return [], nn.Linear(input_dim, num_classes)
        hide = []
        hide.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            hide.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        final = nn.Linear(hidden_dims[-1], num_classes)
        return hide, final
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):

        """Fits the training data on to the target

        This function adjusts the weights according to the data values, so that
        a higher value of accuracy can be achieved. This done for a set number
        of epochs. Nothing is returned.

        Args:
        X (torch.Tensor): _description_
        y (torch.Tensor): _description_
        """

        self.hidden, self.out = self.layers(X.shape[1], self.params_dict['hidden_units'], y.shape[1])
        loss_fn = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=self.params_dict['learning_rate'])
        for epoch in range(self.params_dict['epochs']):
            y_pred = self.forward(X)
            loss = loss_fn(y, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | loss: {loss:.6f}")

    def predict(self, X: torch.Tensor) -> np.ndarray:

        """Predicts the output value

        This function intakes the testing set of data, and then predicts the values
        of the output based on the prevoiusly fitted model on the training set.

        Args:
        X (torch.Tensor): _description_

        Returns:
        np.ndarray: _description_
        """

        predictions = self.forward(X)
        return predictions.detach().numpy()
    
    def get_params(self) -> dict:

        """Returns default hyperparameter values

        This function just returns the names of of the hyperparameters that we will
        be analyzing, and their default values.

        Returns:
        dict: _description_
        """

        dict = {'activation': F.relu, 'hidden_units' : [], 'dropout_p': 0.0, 'learning_rate': 1e-2, 'epochs': 500}
        return dict
    
    def set_params(self, **params: dict) -> dict:

        """Updates the dictionary parameters

        This function takes the hyperparameter values that Optuna recommended for
        the next trial, and sets these values into the dictionary.

        Returns:
        dict: _description_
        """

        for key in params:
            if params[key] == "F.relu":
                self.params_dict[key] = F.relu
            elif params[key] == "F.sigmoid":
                self.params_dict[key] = F.sigmoid
            elif params[key] == "F.tanh":
                self.params_dict[key] = F.tanh
            else:
                self.params_dict[key] = params[key]
        for i in range(self.params_dict["num_layers"]):
            self.params_dict['hidden_units'].append(params[f'n_units_layer{i+1}'])
        return self.params_dict

class ConvolutionalNeuralClassifier(nn.Module):

    def __init__(self):
        
        super(ConvolutionalNeuralClassifier, self).__init__()
        self.params_dict = self.get_params()

    def forward(self, x_in):

        x = self.cnn_layers(x_in)
        x = x.view(x.size(0), -1)
        x_out = self.linear_layers(x)
        return x_out
    
    def layers(self, input_dim, num_classes):

        cnn_layers = nn.Sequential(
            nn.Conv2d(input_dim, self.params_dict['features'], kernel_size=self.params_dict['kernel_size_conv'], stride=self.params_dict['stride_conv'], padding=self.params_dict['padding']),
            nn.BatchNorm2d(self.params_dict['features']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.params_dict['kernel_size_pool'], stride=self.params_dict['stride_pool'])
        )
        linear_layers = nn.Sequential(nn.Linear(self.params_dict['features'] * (((-(((-((self.im_dim + 2 *self.params_dict['padding'] - (self.params_dict['kernel_size_conv'] - 1)) // (-1 *self.params_dict['stride_conv']))) - (self.params_dict['kernel_size_pool'] - 1)) // (-1 * self.params_dict['stride_pool']))) ** 2)) , num_classes))
        return cnn_layers, linear_layers
    
    def fit(self, X, y):

        self.in_dim, self.im_dim, self.out_dim = X.shape[1], X.shape[2], y.shape[1]
        self.cnn_layers, self.linear_layers = self.layers(self.in_dim, self.out_dim)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=self.params_dict['learning_rate'])
        for epoch in range(self.params_dict['epochs']):
            y_pred = self.forward(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = self.accuracy_fn(y_pred=y_pred, y_true=y)
            print(f"Epoch: {epoch} | loss: {loss:.2f}, accuracy: {accuracy:.1f}")
    
    def predict(self, X):

        predictions = self.forward(X)
        predictions = predictions.argmax(axis=1)
        y_predictions = np.zeros((len(X), self.out_dim))
        i = 0
        for val in predictions:
            y_predictions[i][val] = 1
            i = i + 1
        y_predictions = torch.Tensor(y_predictions)
        return y_predictions
    
    def get_params(self):
    
        dict = {'learning_rate': 1e-2, 'epochs': 500, 'features': 3, 'kernel_size_conv': 2, 'stride_conv': 1, 'padding': 1, 'kernel_size_pool':2, 'stride_pool':2}
        return dict

    def set_params(self, **params):

        for key in params:
            self.params_dict[key] = params[key]
        return self.params_dict

    def accuracy_fn(self, y_pred, y_true):
        
        n_correct = torch.eq(y_pred.argmax(axis=1), y_true.argmax(axis=1)).sum().item()
        accuracy = (n_correct / len(y_pred)) * 100
        return accuracy