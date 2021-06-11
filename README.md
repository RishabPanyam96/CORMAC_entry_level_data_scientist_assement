# Objective

The Center for Medicare and Medicaid Services (CMS) has collected data on medical procedures conducted at hospitals under their supervision. The Objective of this assignment is to use the attributes of the procedure to predict the outcome of the procedure. The outcome of the procedure can either be Success or Failure.

# Attributes Used

For the purpose of this assignment I have assumed the following attributes:
Categorical Attributes:
* Severity_of_condition
* Sex
*Doctors_Education_Level
* Type_of_procedure
Continuous Attributes:
* procedure_length(hours)
* No_of_doctors_working_on_procedure
* no_of_other_staff_working_on_procedure
* age_of_person <br>
I have decided not to use outcome measures in my analysis as they are obtained after the procedure is over and hence can’t be used to predict the outcome. To ensure that my code runs I have also created random data for these attributes  as well as for  the outcome and outcome measures.They are created in the **cms_procedures.py** file.

# Cms_procedures.py file

This file is used to generate random data and create functions to supply data for the model.It has all the relevant functions that are to be used and return the data in a dictionary format.

I created a class called cms_procedures having the following functions:
__init__ function to create the data
get_procedure_attributes to get the attributes for a given procedure id
get_procedure_success to get the result for a given procedure id
get_procedure_outcomes to get post procedure outcomes for a given procedure id


# Train.py file

In this file I followed the following steps:
* Created a function to merge attributes,outcome and results of a procedure for a given procedure id
* Collect data from the functions and store it in a DataFrame
* Subset the variables into attributes and the result.
* Perform Train Test split to evaluate model performance without bias
* Perform one hot encoding for categorical variables in the train and test data
* Scale continuous variables in the train and test data
* Convert the target variable into a binary one
* Evaluated correlations in the continuous data. Did not find any highly correlated variables
* Decided on the Random Forest Model. More about this in the next section.
* Use Random Forest Classifier feature importances to do feature selection. Only selected those variables having feature importance greater than 0.10.
* Used Hyperparameter optimization to find best parameters for the model.
* Trained the model with the optimized parameters and calculated test accuracy.
* Obtained a test accuracy of 0.49
* The model was finally saved as pickle file along with the encoders that were used before

# Random Forest Model
The reasons for using the Random Forest Model are:
* Random Forest Models work well with categorical variables
* Random Forest Models do not have many assumptions. The main assumption is that the sample(i.e. our data) is representative of the population.This assumption is used in many models and cannot be verified form our sample data.
* Random Forest Models are flexible but the increase in variance due the flexibility of decision trees are compensated by the bias introduced by random subset selection of feature selection and bagging that takes place in the model.
* Finally Random Forest is a relatively simple model computationally but has high statistical performance in most cases.

# Main_file.py
The main file consists of the final function which takes input of the attributes as a dictionary and outputs the result.To do so we load the model form the pickle file as well as the encoder from a separate pickle file.
