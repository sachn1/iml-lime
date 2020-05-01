# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:39:38 2020

@author: Sachin Nandakumar

This file serves to provide initial values to all the global variables required for the UI to run.

"""

import dash_html_components as html

#####################################################################################################################################
#   Filenames for saved models, datasets & images used in the UI
#####################################################################################################################################

SCALER = 'pickled_models/Aneurysm_StandardScaler.pkl'
ENCODER = 'pickled_models/ohe.pickle'
SCALED_ENCODED_CSV = 'datasets/StandardScaler_OHE_dataset.csv'
DATA = 'datasets/nonnormalised_notohe.csv'
LEGEND_FILENAME = 'assets/graph_leg1.png'
TOOL_SCREENSHOT_FILENAME = 'assets/screenshot.PNG'




#####################################################################################################################################
#   Global initialization for UI variable controls, texts and graphs
#####################################################################################################################################

KW_GLOBAL = '0.65'
NUM_SAMPLE_GLOBAL = 0
MODEL_GLOBAL = 'XGB'
INSTANCE_GLOBAL = 121
TEXT_GLOBAL = html.P(['Model: XGBoost', html.Br(), 'Accuracy: 70.3', html.Br(), html.Br(),'Best Hyperparameter Settings: ', html.Br(), '  n_estimators: 100', html.Br(), '  eta: 0.3', html.Br(), '  colsample_bytree: 1 ', html.Br(), '  max_depth: 2', html.Br(), '  gamma: 4', html.Br(), '  subsample: 1', html.Br(), '  min_child_weight: 4', html.Br(), '  objective: binary:logistic', html.Br(), '  sketch_eps: 0.5', html.Br(), '  tree_method: approx'])
EXPLAINER_GLOBAL = html.Span(['The model (XGBoost) predicts the instance 121 of the training set as Non-ruptured with probability 0.618. The actual class is Ruptured which means that the model has' , html.Span(' incorrectly ', style={'color': 'red', 'font-weight': 'bold'}), 'classified the instance!' ])
DATA_DICT = {'kw_global':3}
UNIT_DICT = {'Width (W)': 'mm', 'Neck (D)': 'mm', 'Trägergefäß (T1)': 'mm', 'Trägergefäß (T2)': 'mm', 'Max. Höhe (Hmax)': 'mm'}
TYPE_DICT = {'Multipel': {1: 'Ja', 2: 'Nein'}, 'Lokalisation': {1: 'ACA', 2: 'Acom', 3: 'Pericall.', 4: 'ACI', 5: '5', 6: 'MCA', 7: 'Pcom',
             8: 'Basilaris', 9: 'PICA', 10: 'Verteb.', 11: 'sonstige'}, 'Seite': {1: 'li', 2: 're', 3: 'mittig'}}
DEGREE_SIGN = u"\N{DEGREE SIGN}"

#####################################################################################################################################