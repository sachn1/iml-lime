# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:00:56 2020

@author: Sachin Nandakumar

This file serves to provide the structure, design and functionality of RUSKLAINER web application

"""


#####################################################################################################################################
#   Import Libraries
#####################################################################################################################################

import dash_gif_component as gif
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import base64
from plot_creation import Plot_Creator
import logging
import flask
import random
import copy
import global_vars

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#####################################################################################################################################
#   Configuration of dash app
#####################################################################################################################################

external_stylesheets =  ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
app.config['suppress_callback_exceptions'] = True

#####################################################################################################################################
#   Initialization of global variables
#####################################################################################################################################
legend_img = global_vars.LEGEND_FILENAME
tool_screenshot = global_vars.TOOL_SCREENSHOT_FILENAME
encoded_legend_image = base64.b64encode(open(legend_img, 'rb').read())
encoded_tool_screenshot = base64.b64encode(open(tool_screenshot, 'rb').read())

fig = go.Figure()

kw_global = global_vars.KW_GLOBAL
num_sample_global = global_vars.NUM_SAMPLE_GLOBAL
model_global = global_vars.MODEL_GLOBAL
instance_global = global_vars.INSTANCE_GLOBAL
text_global = global_vars.TEXT_GLOBAL
explainer_global = global_vars.EXPLAINER_GLOBAL
data_dict = global_vars.DATA_DICT

plot_data = Plot_Creator()
score_data = plot_data.data_for_chart1(model_global, kw_global, num_sample_global, instance_global)
value_data = plot_data.data_for_chart2(model_global, kw_global, num_sample_global, instance_global)
dataframe_for_feature_value_table = plot_data.data_for_table(model_global, kw_global, num_sample_global, instance_global)

#####################################################################################################################################
#   App Layout
#####################################################################################################################################

app.title = 'Rusklainer'
app.layout = html.Div([
        html.Div(
                ############################################
                #   Web Page Banner: GIF + Title
                ############################################
                className='banner',
                children=[
                    html.Div(
                            className='container scalable',
                            children = [
                                    html.Div([
                                     html.H6(
                                        id='banner-title',
                                        children=html.Span([html.Span(children=[gif.GifPlayer(gif='assets/gif2.gif',still='assets/gif2.gif')], className='gif_player playing'), html.Span('RUSKLAINER', style={'fontSize': 35, 'fontWeight': 'bold'}), html.Span(' | ', style={'fontSize': 45,}), 'Identification of contributing features towards ' , html.Span('RU', style={'fontWeight': 'bold'}), 'pture ri', html.Span('SK', style={'fontWeight': 'bold'}), ' prediction using ',  
                                                             html.Span('L', style={'fontWeight': 'bold'}), 'ime expl', html.Span('AINER', style={'fontWeight': 'bold'})]),
                                        style={
                                            "color": "black"
                                        }),
                                    ]),
                    ]),
                     
        ]),
        ############################################
        #   Initialize Web Page Tabs
        ############################################
        dcc.Tabs(
                id="tabs-with-classes",
                value='tab-1',
                parent_className='custom-tabs',
                className='custom-tabs-container',
                children=[
                        dcc.Tab(
                                label='Home',
                                value='tab-1',
                                className='custom-tab',
                                selected_className='custom-tab--selected'
                                ),
                        dcc.Tab(
                                label='Help',
                                value='tab-2',
                                className='custom-tab',
                                selected_className='custom-tab--selected',
                                ),
                         ], colors={
#                        "border": "white",
#                        "background": "black"
                        }),
        html.Div(id='tabs-content-classes'),
        ])

#####################################################################################################################################
#   Tab-wise callback function
#####################################################################################################################################

@app.callback(Output('tabs-content-classes', 'children'),
              [Input('tabs-with-classes', 'value')])
def render_content(tab):
    ############################################
    #   Tab 1: 'Home'
    ############################################
    if tab == 'tab-1':
        return html.Div([
                html.Div([
                        dcc.Loading(
                                id="loading-1",
                                children=[
                                        html.Div([
                                        html.Div([
                                                # hack
                                                html.Div(id='model', style={'display':'none'}),
                                                
                                                ############################################
                                                #   Dropdown for selecting ML models
                                                ############################################
                                                html.Label('Select ML model'),
                                                dcc.Dropdown(
                                                        id='model_dropdown',
                                                        options=[
                                                                {'label': 'XGBoost', 'value': 'XGB',},
                                                                {'label': 'Support Vector Machine', 'value': 'SVM'},
                                                                {'label': 'Random Forest', 'value': 'RForest'}
                                                                ],
                                                        value= model_global,
                                                        searchable=False,
                                                        ),
                                                html.Br(),
                                                # hack
                                                html.Div(id='instances', style={'display':'none'}),
                                                
                                                ############################################
                                                #   Input box for selecting instance number
                                                ############################################
                                                html.Label('Select instance from train set'),
                                                dcc.Input(
                                                        id='inp',
                                                        type="number",
                                                        max=350,
                                                        placeholder="number b/w 0 & 350",
                                                        value= instance_global,
                                                        debounce=True,
                                                        style={'height': 35, 'width': '100%'}
                                                        ),
                                                html.Br(),
                                                html.Br(),
                                                html.H6(
                                                        children='LIME Controls',
                                                        style={
                                                                'textAlign': 'center',
                                                                },
                                                        ),
                                                # hack
                                                html.Div(id='kw_update', style={'display':'none'}),
                                                ############################################
                                                #   Dropdown for selecting kernel width
                                                ############################################
                                                html.Label('Select kernel width'),
                                                dcc.Dropdown(
                                                        id='kw_dropdown',
                                                        options=[
                                                                {'label': '0.40', 'value': 0.40},
                                                                {'label': '0.55', 'value': 0.55},
                                                                {'label': '0.60', 'value': 0.60},
                                                                {'label': '0.65', 'value': 0.65},
                                                                {'label': '2.0', 'value': 2.0}
                                                                ],
                                                        value= kw_global,
                                                        searchable=False,
                                                        ),
                                                html.Br(),
                                                # hack
                                                html.Div(id='sample size slider', style={'display':'none'}),
                                                ############################################
                                                #   Slider for selecting sample size
                                                ############################################
                                                html.Label('Select number of samples'),
                                                dcc.Slider(
                                                        id='sample_slider',
                                                        min=0,
                                                        max=2,
                                                        step=1,
                                                        marks={
                                                                0: '5000',
                                                                1: '10000',
                                                                2: '15000'
                                                        },
                                                        value=num_sample_global,
                                                ),
                                                ], className='pretty_container'),
                                                ############################################
                                                #   Information display for model performance
                                                ############################################
                                                html.Div([
                                                html.H6(
                                                        children='Performance of Model',
                                                        style={
                                                                'textAlign': 'center',
                                                                },
                                                        ),
                                                html.Div([
                                                        html.P(id='my_text',children=text_global)
                                                    ]),
                                                html.Br(),
                                                # hack
                                                html.Div(id='performance_update', style={'display':'none'}),
                                                html.Div(id='target'),
                                                ], className='pretty_container'),
                                            ], className='two columns', style={'width':'20%'}),
                                        
                                        ############################################
                                        #   Feature Importance Graph
                                        ############################################
                                        
                                        html.Div([
                                                html.Div([
                                                html.Div([
                                                        html.H4('LIME Explanation'),
                                                        dcc.Markdown('''
                                                                     ---------------------------------------
                                                                     '''),
                                                        html.Img(src='data:image/png;base64,{}'.format(encoded_legend_image.decode()), style={'width':'45%','display': 'inline-block'}),
                                                        ], style={'textAlign': 'center'}),
                                                        dcc.Graph(
                                                                id='LIME',
                                                                figure = {'data': score_data,
                                                                          'layout': {
                                                                                  'height': '55%',
                                                                                  'margin': {
                                                                                          'l': 200, 'b': 20, 't': 30, 'r': 30
                                                                                          },
                                                                                          'xaxis' : {'range':[-.25,.25], 'title':'Feature Effect: Coefficients of LIME explanations', 'automargin': True}, 
                                                                                          'yaxis' : dict(autorange="reversed"),
                                                                                          'paper_bgcolor' : 'rgba(0,0,0,0)',
                                                                                          'plot_bgcolor' : 'rgba(0,0,0,0)'
                                                                                    }
                                                                          }
                                                        ),
                                                html.Br(),
                                                html.Div([
                                                        dcc.Markdown('''
                                                                     ** The above legend indicates the contribution of each feature to the respective classes. Subsequently, it also indicates how much each feature supports/contradicts the black box prediction.
                                                                     '''),
                                                        ], style={'fontSize':14, 'textAlign': 'center'}),
                                                ], className='pretty_container'),
                                                html.Div([
                                                        html.P(id='explainer text',children=[explainer_global])
                                                    ], style={'fontSize':20, 'textAlign': 'center'}),
                                                    html.Div(id='target2'),
                                                ], className='seven columns',),
                                                
                                                 ############################################
                                                #   Black box prediction prob. graph
                                                ############################################
                                                html.Div([
                                                        html.Div([
                                                                html.H6(children='Black-box prediction probabilities'),
                                                                dcc.Graph(
                                                                        id='prediction probability',
                                                                        figure={
                                                                        'data': [go.Bar( x = [round(value_data[2],3), round(value_data[1],3)],
                                                                                              y = ['Non-ruptured', 'Ruptured'],
                                                                                              marker= dict(
                                                                                                      color=['#004c00', '#990000']
                                                                                            ),
                                                                                              orientation='h')], 
                                                                        'layout': {
                                                                                'height': 110,
                                                                                'margin': {'l': 82, 'b': 25, 't': 30, 'r': 10},
                                                                                'xaxis': {'range': [0, 1], 'title':'Prediction Probability of Rupture status', 'automargin': True}, 
                                                                                'paper_bgcolor' : 'rgba(0,0,0,0)',
                                                                                'plot_bgcolor' : 'rgba(0,0,0,0)'
                                                                                }
                                                                        }
                                                                ),
                                                        ], className='pretty_container'),
                                                ############################################
                                                #   Feature value display of instance
                                                ############################################
                                                html.Div([
                                                html.H6(id='feature-label', children='Feature values of Instance {}'.format(instance_global)),
                                                dash_table.DataTable(
                                                        id='feature table',
                                                        columns=[{"name": i, "id": i} for i in dataframe_for_feature_value_table.iloc[:, : 2].columns],
                                                        data=dataframe_for_feature_value_table.to_dict("rows"),
                                                        style_table={
                                                                'maxHeight': '100%',
                                                                'width': '100%',
                                                                'minWidth': '100%',
                                                        },
                                                        style_cell={
                                                                'fontFamily': 'sans-serif',
                                                                'textAlign': 'left',
                                                                'whiteSpace': 'inherit',
                                                                'overflow': 'hidden',
                                                                'textOverflow': 'ellipsis',
                                                        },
                                                        style_header={
                                                                'backgroundColor': 'rgb(230, 230, 230)',
                                                                'fontWeight': 'bold'
                                                        },  
                                                        style_data_conditional=[
                                                                {
                                                                        'if': {'row_index': 'odd'},
                                                                        'backgroundColor': 'rgb(248, 248, 248)'
                                                                }
                                                        ]
                                                ),
                                                ], className='pretty_container'),

                                                ############################################
                                                #   App submission info.
                                                ############################################
                                                html.Div(
                                                        dcc.Markdown('''
                                                                     *This application is part of the following submission:* \n
                                                                     *Identification of Features that Contribute towards Rupture Risk Prediction in Intracranial Aneurysms using LIME*\n
                                                                     *Sachin Nandakumar, Yash Shah, Uli Neimann, Sylvia Saalfeld*                            
                                                                     '''),
                                                        style={
                                                                'display': 'inline-block',
                                                                'fontSize': 11,
                                                                'textAlign': 'right',
                                                                #'color':'white'
                                                        },
                                                ),
                                                html.Br(),
                                                html.Div(id='target3'),
                                ], className='three columns'),
                    ], type=random.choice(['graph','cube']), debug=True)], className="row"),
                
            ])
    ############################################
    #   Tab 2: 'Help'
    ############################################
    elif tab == 'tab-2':
        return html.Div([
                        html.Div([
                                html.Br(),
                                html.Div(
                                        children = [
                                                dcc.Markdown('''
                                                             
                                                             ** LIME aims to create local models – one for each observation of the dataset to be explained. These local models operate in the neighbourhood of the instance (of the dataset) to be explained. **
                            
                                                            Given below are a step-by-step video tutorial and detailed description of this application. It should give you a better understanding of what each component represents.                   
                                                            '''),
                                                ############################################
                                                #   Video tutorial iframe
                                                ############################################
                                                html.Div(
                                                        children=[
                                                                html.Iframe(srcDoc='<video controls width="1500"> <source src="/assets/vid.mp4" type="video/mp4"> Sorry, your browser doesn\'t support embedded videos.</video>',
                                                                            width="1600", height="900", style={'border' : 0}),
                                                                            ],
                                                                style={
                                                                        'vertical-align' : 'middle',
                                                                        'horizontal-align' : 'middle',
                                                                        'textAlign': 'center'
                                                                        
                                                                },
                                                        ),
                                                
                                        ],
                                        style={
                                                    'fontSize': 18,
                                                    'textAlign': 'center'
                                                    },
                                ),  
                        ], className='one columns' , style={'width': '100%'}),
            
                    html.Div([
                        html.Div(
                                children = [
                                ############################################
                                #   Descriptive screenshot of tool
                                ############################################
                                html.Div(
                                        children=[
                                        html.Img(src='data:image/png;base64,{}'.format(encoded_tool_screenshot.decode()), style={'width': '100%',}),
                                ], className="six columns"),
                                
                                ############################################
                                #   Tool description
                                ############################################
                                html.Div(
                                        children=[
                                    html.H5('Description'),
                                    dcc.Markdown('''
                                                 
                                                 **1)** Select model - This dropdown is provided with 3 black-box models (XGBoost, Support Vector Machine & Random Forest) having the best performance evaluated against other 5 models using nested cross-validation.\n
                                                 **2)** Select instance - Each instance corresponds to the aneurysm data of a patient in the dataset. The LIME model explains the black box prediction of the selected instance. Since the training dataset consists of 351 patients, this input box accepts values from 0 to 350.\n
                                                 *Select variables for LIME prediction:* \n
                                                 **3)** The kernel width determines how large the neighbourhood is: A small kernel width means that an instance must be very close to influence the local model, a larger kernel width means that instances that are farther away also influence the model. However, there is no best way to determine the width. The chosen kernel width of **0.65** was where the surrogate model was found to converge locally before global convergence when inspected over a range of values to find stable locality. \n
                                                 **4)** The sample size allows you to choose the number of samples to be perturbed around the instance of interest **(2)** within the kernel width **(3)**.\n
                                                 *Hence LIME model draws **s** number of samples from a normal distribution around the instance **i** within the kernel width **k**. * \n
                                                 **5)** Displays the performance of the selected model **(1)** and the corresponding hyperparameter settings chosen for the model using cross-validation.\n
                                                 **6)** This graph shows the contribution of features towards the black-box model's **(1)** prediction of rupture status of aneurysm of the selected instance **(2)**. The importance/effect of a feature is evaluated by its coefficients in the LIME model (ridge regression). The features are sorted based on the importance of the predicted class.\n
                                                 **7)** Displays the information on whether the instance is predicted correctly/incorrectly by the black-box model on comparison with the actual rupture status from the train set.\n
                                                 **8)** This graph shows the black-box model's prediction probabilities of rupture status of aneurysm of the instance\n
                                                 **9)** Displays the original value of each feature of the instance selected **(2)**. The values of categorical features (Multipel, Lokalisation & Seite) are encoded to numbers before running the machine learning models. Displayed are the encoded as well as the real values of those features.
                                                     '''),
                                ], className="six columns"),
                            ], className='pretty_container twelve columns', style={'fontSize': 13},),
                                                 
# alternate layout. screenshot on another (right hand side) container - need not remove.
#                        html.Div([
#                                html.Div(
#                                        children=[
#                                        html.H6('Reference Screenshot'),
#                                        html.Img(src='data:image/png;base64,{}'.format(encoded_tool_screenshot.decode()), style={'width': '100%', 'height': '100%'}),
#                                ], style={'vertical-align' : 'middle',
#                                        'horizontal-align' : 'middle'
#                                                },),
#                            ], className='pretty_container six columns')
                    ]),
                    
        ]),
                    
                    

#####################################################################################################################################
#
#   CALLBACK FUNCTIONS OF EACH COMPONENT
#
#####################################################################################################################################
                                 
# black-box model dropdown callback
@app.callback(Output('model', 'children'),
              [Input('model_dropdown', 'value')])
def handle_dropdown_value(model_dropdown):
    return model_dropdown

# instance number input box callback
@app.callback(Output('instances', 'children'),
              [Input('inp', 'value')])
def update_instance_value(value):
    if value is not None:
        return value
    else:
        return 0
    
# kernel width dropdown callback
@app.callback(Output('kw_update', 'children'),
              [Input('kw_dropdown', 'value')])
def update_kw_value(value):
    return value

# sample size slider callback
@app.callback(Output('sample size slider', 'children'),
              [Input('sample_slider', 'value')])
def update_sample_value(value):
    return value

# model performance info text callback
@app.callback(Output('my_text', 'children'),
              [Input('model_dropdown', 'value')])
def update_parameter_text(model_dropdown):
    global text_global
    text_global = plot_data.update_parameter_text(model_dropdown) 
    return text_global

# explainer text callback
@app.callback(Output('explainer text', 'children'),
              [Input('model_dropdown', 'value'), Input('inp', 'value'), Input('kw_dropdown', 'value'), Input('sample_slider', 'value')])
def update_explainer_text(model_dropdown, inp, kw_dropdown, sample_slider):
#    global explainer_global
    if inp is None or inp == 0:
        inp = 0
    text = plot_data.update_explainer_text(model_dropdown, kw_dropdown, sample_slider, inp) 
    if ' correctly' in text:
        text1, text2 = text.split('correctly')
        return html.Span([text1 , html.Span(' correctly ', style={'color': 'green', 'fontWeight': 'bold'}), text2])
    else:
        text1, text2 = text.split('incorrectly')
        return html.Span([text1 , html.Span(' incorrectly ', style={'color': 'red', 'fontWeight': 'bold'}), text2])

# feature importance graph callback
@app.callback(
        Output('LIME', 'figure'), 
        [Input('model_dropdown', 'value'), Input('inp', 'value'), Input('kw_dropdown', 'value'), Input('sample_slider', 'value')])
def callback1(model_dropdown, inp, kw_dropdown, sample_slider):
    if inp is None or inp == 0:
        inp = 0
    return {'data': plot_data.data_for_chart1(model_dropdown, kw_dropdown, sample_slider, inp),
            'layout': {
                    'height': '50%',
                    'margin': {
                            'l': 200, 'b': 20, 't': 30, 'r': 30
                            },
                            'xaxis' : {'range': [-.25,.25], 'title':'Feature Effect: Coefficients of LIME explanations', 'automargin':True},
                            'yaxis' : dict(autorange="reversed"),
                            'paper_bgcolor' : 'rgba(0,0,0,0)',
                            'plot_bgcolor' : 'rgba(0,0,0,0)'
                }
        }    
        
        
def global_store(model_dropdown, kw_dropdown, sample_slider, inp):
    value_data = plot_data.data_for_chart2(model_dropdown, kw_dropdown, sample_slider, inp)
    return value_data

def generate_figure(model_dropdown, kw_dropdown, sample_slider, inp, figure):
    fig = copy.deepcopy(figure)
    value_data = global_store(model_dropdown, kw_dropdown, sample_slider, inp)
    fig['data'][0]['x'] = [round(value_data[2],3), round(value_data[1],3)]
    fig['layout'] = {'height': 110, 'margin': {'l': 82, 'b': 25, 't': 30, 'r': 10}, 'xaxis': {'range': [0, 1], 'title':'Prediction Probability of Rupture status', 'automargin': True}, 'paper_bgcolor' : 'rgba(0,0,0,0)',  'plot_bgcolor' : 'rgba(0,0,0,0)'}
    return fig
    
# black-box prediction prob graph callback                    
@app.callback(
        Output('prediction probability', 'figure'), 
        [ Input('model_dropdown', 'value'), Input('inp', 'value'), Input('kw_dropdown', 'value'), Input('sample_slider', 'value')])
def callback2(model_dropdown, inp, kw_dropdown, sample_slider):
    global value_data, instance_global
    if inp is None or inp == 0:
        inp = 0
    
    return generate_figure(model_dropdown, kw_dropdown, sample_slider, inp, {'data': [
                        go.Bar( 
                                y = ['Non-ruptured', 'Ruptured'],
                                marker= dict(
                                        color=['#004c00', '#990000']
                                ),
                                orientation='h'
                        )],
            })    
        
# feature value display table label callback
@app.callback(Output('feature-label', 'children'),
              [Input('inp', 'value')])
def update_label(value):
    if value is not None:
        return 'Feature values of Instance {}'.format(value)
    else:
        return 'Feature values of Instance {}'.format(0)        
        
# feature value display table callback
@app.callback(Output('feature table', 'data'), 
              [Input('model_dropdown', 'value'), Input('inp', 'value'), Input('kw_dropdown', 'value'), Input('sample_slider', 'value')])
def update_table(model_dropdown, inp, kw_dropdown, sample_slider):
    global dataframe_for_feature_value_table, instance_global
    if inp is None or inp == 0:
        inp = 0
    dataframe_for_feature_value_table = plot_data.data_for_table(model_dropdown, kw_dropdown, sample_slider, inp)
    return dataframe_for_feature_value_table.to_dict('records')    
        
# main()
if __name__ == '__main__':
    app.run_server(debug=True)