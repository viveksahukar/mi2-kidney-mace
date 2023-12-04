import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn_pandas import DataFrameMapper
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW, SGD, Adam, Adagrad, RMSprop
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CyclicLR, StepLR, MultiStepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import torchtuples as tt
from datetime import timedelta
import time
from roc_utils import *
import pickle
import normalizer
import utils
import random
import json
from torchsampler import ImbalancedDatasetSampler
from sklearn.utils import resample
import tqdm
import plotly
from plotly import __version__ as plotly_version
# with try_import() as _imports:
#     import plotly
#     from plotly import __version__ as plotly_version


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, brier_score_loss, fbeta_score, roc_curve, average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
import imblearn
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from IPython.display import display

import ast

from torch.utils.tensorboard import SummaryWriter

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

