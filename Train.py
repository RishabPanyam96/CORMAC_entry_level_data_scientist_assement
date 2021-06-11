# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 21:21:07 2021

@author: 13129
"""
#import packages
import cms_procedures
import pickle
import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 500)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# Define the class from which to generate the data
CMS_data_class = cms_procedures.cms_procedures(100)

# Function to merge attributes,result and outcome data into single dictionary 
def get_merged_data(procedure_id,CMS_data_class):
    dict_final = CMS_data_class.get_procedure_attributes(procedure_id)
    dict_outcomes = CMS_data_class.get_procedure_outcomes(procedure_id)
    dict_final.update(dict_outcomes)
    dict_final['Result'] = CMS_data_class.get_procedure_success(procedure_id)
    dict_final['procedure_id'] = procedure_id
    return dict_final

# Function to create Final Dataframe

def get_dataframe_from_data(CMS_data_class):
    dict_return = {}
    counter = 1
    while True :
        try : 
            dict_return[counter] = get_merged_data(counter,CMS_data_class)
            counter+=1
        except KeyError:
            print(f'Total no of rows is {counter-1}')
            break
    DF= pd.DataFrame.from_dict(dict_return,orient='index')     
    return DF

#Get Final Dataframe

Total_Dataframe = get_dataframe_from_data(CMS_data_class)

# define categorical and continous variables

categorical_variables = ['severity_of_condition','sex','Doctors_Education_Level','Type_of_procedure']
continous_variables   = ['procedure_length','no_of_doctors_working_on_procedure','no_of_other_staff_working_on_procedure','age_of_patient'] 

# subset the data to attributes and target variable 
def get_final_df(DF):
    return DF[categorical_variables + continous_variables+['Result']]

final_df = get_final_df(Total_Dataframe)

print(final_df[continous_variables].corr())
print('We can see that none of the variables have high values of correlation and hence no issues of collinearity in the data' )

# Function to split the Data into test and train sets 

def Train_test_split(DF,testsize =0.25,seed=100):
    X= DF[categorical_variables+continous_variables] 
    y= DF['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=seed)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = Train_test_split(final_df)
# one hot encoding categorical variables 

def get_one_hot_encoded_data(X_train,X_test,cat_var):
    one_hot_encoder = OneHotEncoder(handle_unknown= 'ignore').fit(X_train[cat_var])
    with open('one_hot_encoder.pkl', 'wb') as f:
         pickle.dump(one_hot_encoder,f)
    X_train_encoded = pd.DataFrame(one_hot_encoder.transform(X_train[cat_var]).toarray(),columns = one_hot_encoder.get_feature_names(cat_var))
    X_test_encoded = pd.DataFrame(one_hot_encoder.transform(X_test[cat_var]).toarray(),columns = one_hot_encoder.get_feature_names(cat_var))
    return X_train_encoded,X_test_encoded

# scaling continous varaibles
def scale_continous_data(X_train,X_test,cont_var):
    scaler = StandardScaler()
    scaler.fit(X_train[cont_var])
    with open('standardscaler.pkl','wb') as f:
         pickle.dump(scaler,f)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train[cont_var]),columns=cont_var)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[cont_var]),columns= cont_var)
    return X_train_scaled,X_test_scaled

# converting target variable into a binary (0,1) variable
def target_variable_encoding(y_train,y_test):
    y_train_ret = y_train.apply(lambda x : 1 if x == 'Success' else 0)
    y_test_ret = y_test.apply(lambda x : 1 if x == 'Success' else 0)
    return y_train_ret,y_test_ret

# Applying one hot encoding and varaible scaling
def data_preprocessing(df,cat_var,cont_var):
    X_train, X_test, y_train, y_test = Train_test_split(df)
    X_train_encoded,X_test_encoded = get_one_hot_encoded_data(X_train, X_test, cat_var)
    X_train_scaled,X_test_scaled = scale_continous_data(X_train, X_test, cont_var)
    y_train_ret,y_test_ret = target_variable_encoding(y_train, y_test)
    X_train_ret = pd.concat([X_train_encoded,X_train_scaled],axis=1)
    X_test_ret = pd.concat([X_test_encoded,X_test_scaled],axis = 1)
    return X_train_ret,X_test_ret,y_train_ret,y_test_ret

X_train_ret,X_test_ret,y_train_ret,y_test_ret = data_preprocessing(final_df,categorical_variables,continous_variables)

# Feature Selection based on feature importances
def select_important_features(X_train_ret,y_train_ret):
    clf = RandomForestClassifier()
    sfm =SelectFromModel(clf,threshold =0.10)
    sfm.fit(X_train_ret, y_train_ret)
    feature_labels = X_train_ret.columns
    for feature_list_index in sfm.get_support(indices=True):
        print(feature_labels[feature_list_index])
    return sfm    
#Transform the data to include only important variables
sfm = select_important_features(X_train_ret, y_train_ret)
X_important_train = sfm.transform(X_train_ret)
X_important_test = sfm.transform(X_test_ret)    

#Hyperparamter optimization using GridSearchCV

def get_best_hyperparamters(X_train_ret,y_train_ret):
     
    param_grid = {
    'bootstrap': [True],
    'max_depth': [40,70, 100],
    'max_features':[1,2,3], 
    'min_samples_leaf': [3,5,],
    'min_samples_split': [5, 10],
    'n_estimators': [100, 200, 1000]
     }
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 1,return_train_score = True)
    grid_search.fit(X_train_ret,y_train_ret)
    print(grid_search.best_params_)
    return grid_search.best_params_

# Fitting model on the train data with best hyperparameters
best_params = get_best_hyperparamters(X_train_ret, y_train_ret)
clf = RandomForestClassifier()
clf.set_params(**best_params)
clf.fit(X_important_train,y_train_ret)

print(y_train_ret)

# predict for the test set
y_pred = clf.predict(X_important_test)
print(y_pred)
print(f'The accuracy of the model is {accuracy_score(y_test_ret, y_pred)}')

with open('Random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

        
    
    