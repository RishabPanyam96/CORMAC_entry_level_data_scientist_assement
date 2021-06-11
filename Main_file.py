# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:25:01 2021

@author: 13129
"""
# import packages
import pickle
import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 500)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#test dict
dict_input = {'severity of condition':'High','sex':'Male','Doctors_Education_Level':'Bachelors','Type_of_procedure':'Heart','procedure_length':2,
              'no_of_doctors_working_on_procedure':10,'no_of_other_staff_working_on_procedure':6,'age_of_patient':45}

# prediction function                                    
def get_prediction(dict_input):
    #loading scaler
    pkl_file = open('standardscaler.pkl', 'rb')
    scaler= pickle.load(pkl_file) 
    pkl_file.close()
    Df= pd.DataFrame(dict_input,index=[0])
    # subsetting only the relevant varaibles
    Df = Df[['procedure_length','no_of_doctors_working_on_procedure','no_of_other_staff_working_on_procedure','age_of_patient']]
    Df = scaler.transform(Df)
    pkl_file = open('Random_forest_classifier.pkl', 'rb')
    clf= pickle.load(pkl_file) 
    pkl_file.close()
    y = clf.predict(Df)[0]
    if y== 0: 
        return 'Failure'
    else :
        return 'Success'

print(get_prediction(dict_input))
    