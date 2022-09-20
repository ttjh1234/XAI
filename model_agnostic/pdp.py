# Partial Dependence Plot (PDP)

'''

PDP shows the marginal effect one or two features have on the predicted outcome of a ml model. (J.H. Friedman 2001)

'''

import pandas as pd
import sklearn
import numpy as np
import os
from data import data_preprocess


data=data_preprocess(path)


