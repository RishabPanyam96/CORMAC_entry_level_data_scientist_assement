# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:10:16 2021

@author: Rishab Panyam
"""

# Loading Packages

import numpy as np
import pandas as pd

# Creating Random Data For the model

# Continous features generated : procedure_length(hours),no_of_doctors_working_on_procedure,no_of_other_staff_working_on_procedure,age_of_person
# Categorical features generated : severity_of_condition,sex,Doctors_Education_Level,Type_of_procedure
# Outcome features generated: severity_of_post_procedure_complications,recurrence_of_orignal_condition 
class cms_procedures:
    
    def __init__(self,seed):
        np.random.seed(seed)
        self.rows = np.random.randint(1000,10000,1,)[0]
        dict_attributes = {}
        dict_results = {}
        dict_outcomes = {}
        # Define Possible Values for the variables
        severity_of_conditions_list = ['High','Medium','Low']
        sex_list = ['Male','Female']
        Doctors_Education_Level_list = ['Diploma','Bachelors','Higher Studies','Other']
        Type_of_procedure_list = ['Heart','Brain','Stomach','Leg','Back','Kidney','Liver']
        severity_of_post_procedure_complications_list = ['High','Medium','Low','No complications']
        recurrence_of_orignal_condition_list = ['True','False']
        outcome_of_procedure_list = ['Success','Failure']
        for i in range(self.rows):
            procedure_id = i+1
            severity_of_condition = np.random.choice(severity_of_conditions_list,1)[0]
            sex = np.random.choice(sex_list,1)[0]
            Doctors_Education_Level = np.random.choice(Doctors_Education_Level_list,1)[0]
            Type_of_procedure = np.random.choice(Type_of_procedure_list,1)[0]
            # Random Catgorical Values in their respective ranges
            procedure_length = np.random.randint(2,48,1)[0]
            no_of_doctors_working_on_procedure = np.random.randint(2,8,1)[0]
            no_of_other_staff_working_on_procedure = np.random.randint(3,10,1)[0]
            age_of_patient = np.random.randint(0,90,1)[0]
            # Target Variable
            outcome_of_procedure = np.random.choice(outcome_of_procedure_list,1)[0]
            # Outcome Measures
            severity_of_post_procedure_complications = np.random.choice(severity_of_post_procedure_complications_list,1)[0]
            if outcome_of_procedure == 'Success':
               recurrence_of_orignal_condition = np.random.choice(recurrence_of_orignal_condition_list,1)[0] 
            else :
               recurrence_of_orignal_condition = None                                
            # We generate a dictionary of dictionaries with the first key being the porcedure_id while the inner dictionaries keys being the measures.
            dict_attributes_temp = {'severity_of_condition':severity_of_condition,'sex':sex,'Doctors_Education_Level':Doctors_Education_Level,
                                    'Type_of_procedure':Type_of_procedure,'procedure_length':procedure_length,
                                    'no_of_doctors_working_on_procedure':no_of_doctors_working_on_procedure,
                                    'no_of_other_staff_working_on_procedure':no_of_other_staff_working_on_procedure,
                                    'age_of_patient':age_of_patient}
            dict_attributes[procedure_id] = dict_attributes_temp
            dict_results[procedure_id] = outcome_of_procedure
            dict_outcomes_temp = {'severity_of_post_procedure_complications_list':severity_of_post_procedure_complications,'recurrence_of_orignal_condition':recurrence_of_orignal_condition}
            dict_outcomes[procedure_id] = dict_outcomes_temp
       
            
        self.dict_attributes = dict_attributes
        self.dict_results = dict_results
        self.dict_outcomes = dict_outcomes
            
    def get_procedure_attributes(self,procedure_id = None):
        if procedure_id == None:
            return self.dict_attributes[np.random.randint(0,self.rows)]
        else: 
            return self.dict_attributes[procedure_id]
        
    def get_procedure_success(self,procedure_id):
        return self.dict_results[procedure_id]
    
    def get_procedure_outcomes(self,procedure_id):
        return self.dict_outcomes[procedure_id]



temp_data_class = cms_procedures(100)


            
            
            
         