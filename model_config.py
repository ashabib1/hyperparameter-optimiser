from typing import Literal
from pydantic import BaseModel, confloat, conint

class RandomForestMP(BaseModel):
    n_estimators: conint(gt=1, le=200) = 100
    criterion: Literal["squared_error", "absolute_error", "poisson"] = "squared_error"
    max_depth: conint(gt=1, le=100) = None
    min_samples_split: conint(gt=2, le=10) = 2
    min_samples_leaf: conint(gt=1, le=50) = 1
    min_weight_fraction_leaf: confloat(ge=0.0, le=0.5) = 0.0
    max_features: confloat(ge=0.0, le=1.0) = 1.0
    max_leaf_nodes: conint(gt=2, le=100) = None
    min_impurity_decrease: confloat(ge=0.0, le=10.0) = 0.0
    bootstrap: Literal[True, False] = True
    oob_score: Literal[True, False] = False
    n_jobs: conint(ge=1, le=100) = None
    random_state: conint(ge=1, le=100) = None
    verbose: conint(ge=0, le=100) = 0
    warm_start: Literal[True, False] = False
    ccp_alpha: confloat(ge=0.0, le=1.0) = 0.0
    max_samples: confloat(ge=0.0, le=1.0) = None

class LassoMP(BaseModel):
    alpha: confloat(gt=0, le=2.0) = 1.0
    fit_intercept: Literal[True, False] = True
    precompute: Literal[True, False] = False
    copy_X: Literal[True, False] = False
    max_iter: conint(ge=1000, le=10000) = 1000
    tol: confloat(ge=1e-7, le=1e-2) = 1e-4
    warm_start: Literal[True, False] = False
    positive: Literal[True, False] = False
    random_state: conint(ge=1, le=100) = None
    selection: Literal["cyclic", "random"] = "cyclic"

class NeuralMP(BaseModel):
    activation: Literal["F.relu"]
    num_layers: conint(ge=1, le=5) = 2
    n_units_layer1: conint(ge=50, le=500) = 250
    dropout_p: confloat(ge=0, lt=0.5) = 0
    learning_rate: confloat(gt = 1e-7, lt = 1e-2) = 1e-4
    epochs: conint(ge=10, lt=1000) = 100
    batch_size: conint(ge=10, lt=100) = 32

class ConvolutionalMP(BaseModel):
    learning_rate: confloat(gt = 1e-7, lt = 1e-2) = 1e-4
    epochs: conint(ge=50, le=500) = 250
    features: conint(ge=2, le=5) = 3
    kernel_size_conv: conint(ge=2, le=5) = 3
    stride_conv: conint(ge=1, le=3) = 2
    padding: conint(ge=0, le=2) = 1
    kernel_size_pool: conint(ge=2, le=5) = 3
    stride_pool: conint(ge=1, le=3) = 2