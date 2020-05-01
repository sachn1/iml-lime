"""
@author: Yash Shah

This file is the core file for LIME model. 

"""



#####################################################################################################################################
#   Import Libraries
#####################################################################################################################################

import numpy as np
import pandas as pd
from sklearn import preprocessing 
import pickle
import lime
import lime.lime_tabular
import global_vars


class LIME_Explainer:
    def __init__(self):
        self.encoder = pickle.load(open(global_vars.ENCODER, 'rb'))
        
    def get_data(self, model):
        '''
            This function gets all data required to run the LIME explainer
        '''
        
        filename = 'pickled_models/'+model.lower()+'.pkl'
        model = pickle.load(open(filename, 'rb'))
        
        scaled_encoded = pd.read_csv(global_vars.SCALED_ENCODED_CSV,header=None)
        original_data = pd.read_csv(global_vars.DATA)
        label_col = original_data['Rupturiert']
        original_data = original_data.drop(columns = ['Rupturiert'])
        
        return model, scaled_encoded, original_data, label_col
    
    def preprocess_train(self, model, scaled_encoded, X2, X3):
        '''
            This function does basic preprocessing of the files fetched from get_data 
        '''
        categorical_features1 = [0,1,2]
        categorical_names = {}
        for feature in categorical_features1:
            le = preprocessing.LabelEncoder()
            le.fit(X2.iloc[:, feature])
            X2.iloc[:, feature] = le.transform(X2.iloc[:, feature])
            categorical_names[feature] = le.classes_
            
        le= preprocessing.LabelEncoder()
        le.fit(X3)
        X3 = le.transform(X3)
        class_names = le.classes_
        
        return scaled_encoded, X2, X3, class_names, categorical_names, categorical_features1
    
    
    
    def explainer(self, model, x1, x2, categorical_features1, categorical_names, class_names, instance_no, krw, numsamp):
        '''
            This function runs the tabular explainer for LIME and outputs the scores, attributes and predictions
        '''
        predict_fn = lambda x: model.predict_proba(self.encoder.transform(x)).astype(float)
        
        explainer1 = lime.lime_tabular.LimeTabularExplainer(np.array(x2),feature_names = x2.columns.tolist(),
                                                     class_names=class_names,
                                                    categorical_features=categorical_features1, 
                                                    categorical_names=categorical_names,
                                                    feature_selection = 'forward_selection', 
                                                    sample_around_instance=True,
                                                    discretize_continuous=True, kernel_width=krw,random_state=42) 
        
        
        exp1 = explainer1.explain_instance(x2.iloc[instance_no,:], predict_fn,num_samples = numsamp)
        
        check_list = exp1.as_list()
        colus = x2.columns.tolist()
        temp_frame = []
        for j in range(len(check_list)):
            contained = [x for x in colus if x in check_list[j][0]]
            temp_frame.append(contained[0])
        
        empty = {k:[] for k in temp_frame}
        
        for j in range(len(check_list)):
            contained = [x for x in colus if x in check_list[j][0]]
            empty[contained[0]] = check_list[j][1]
        
        score = {k: v for k, v in sorted(empty.items(), key=lambda item: abs(item[1]),reverse=True)}
        
        store_attribute = dict(x2.iloc[instance_no,:])
        
        val = np.array(x1.iloc[instance_no,:])[np.newaxis]
        preddict = {k:[] for k in [1,2]}
        preddict[1] = model.predict_proba(val)[0][0]
        preddict[2] = model.predict_proba(val)[0][1]
        
        return score, store_attribute, preddict
    
    def run_LIME(self, model, kernel_width, sample_size, instance_number):
        '''
            main()
        '''
        model, processed, anuer, label_col = self.get_data(model)
        processed, anuer, label_col, class_names, categorical_names, categorical_features1 = self.preprocess_train(model, processed, anuer, label_col)
        score_dict, attri_dict, probab_dict = self.explainer(model, processed, anuer, categorical_features1, categorical_names, class_names,
                                                    instance_number, kernel_width, sample_size)
        return score_dict, attri_dict, probab_dict, label_col[instance_number]
     
        
    
    
#####################################################################################################################################
#   Code for Unit-testing
#####################################################################################################################################

# uncomment below code for testing this file

#    def run_LIME_(self, model, kernel_width, sample_size, instance_number):
#        model, processed, anuer, label_col = self.get_data(model)
#        processed, anuer, label_col, class_names, categorical_names, categorical_features1 = self.preprocess_train(model, processed, anuer, label_col)
#        exp1 = self.explainer(model, processed, anuer, categorical_features1, categorical_names, class_names,
#                                                    instance_number, kernel_width, sample_size)
#        return exp1
    
#lime_exp = LIME_Explainer()
#score_dict, attri_dict, probab_dict, x = lime_exp.run_LIME(model='xgb', kernel_width=0.5, sample_size=5000, instance_number=6)
#print(probab_dict)
#exp1 = lime_exp.run_LIME_(model='xgb', kernel_width=0.65, sample_size=5000, instance_number=121)
#
#html = exp1.as_html()
#with open("121.html", "w") as file:
#    file.write(html)