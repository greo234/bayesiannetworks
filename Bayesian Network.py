# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:06:38 2023

Bayesian Network prediction to get which field women are most likely to 
to go for

@author: Omoregbe Olotu
"""

# -*- coding: utf-8 -*-

# importing relevant libraires
# for loading the data
import pandas as pd

# To import Data Visualization libraries
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

# For creating the Bayesian Network model
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

# For making inferences
from pgmpy.inference import VariableElimination

# For DAG visualization
import networkx as nx
from daft import PGM

# To import dataset

names = ["age", "workclass", "fnlwgt", "education", "education_number", "marital_status", 
         "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours-per-week",
         "native-country", "income"]

income_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
                         names=names)

"""### UNDERSTANDING THE DATA"""

# To view the top five rows of data
income_data.head(5)

income_data_shape = income_data.shape
print('This dataset has {} rows and {} columns'.format(income_data_shape[0], income_data_shape[1]))

# To get an overview of the data
income_data.info()

# to gain deeper insights on the numerical values
income_data.describe()

# Checking for null values
income_data.isnull().sum()

"""There are no null values"""

# Checking for duplicates
income_data.duplicated().sum()

"""### DATA CLEANING AND PREPARATION
Using the info() function, I noticed some column names will hypen, this will be replaced with underscores to avoid running into avoidable errors.
"""

# To remove hyphens from column names
income_data.columns = income_data.columns.str.replace('-','_')

# Removal of duplicates
income_data.drop_duplicates(keep = 'first', inplace = True)

"""In the next steps, I will inspect the unique values in the categorical rows. This step will help me to eliminate values that do not comply with the data type of that column."""

# For the workclass column
print(income_data['workclass'].unique())

"""Looking at the uniques values of the workclass data, I can tell that the value '?' does not belong there. I will retrieve all rows with that values for further inspection."""

# To retrieve the affected rows
income_data[income_data['workclass'] == ' ?'].sample(50)

"""Documentation of this dataset revealed that the '?' values was used to fill up missing data. Having established this and also because the number of rows affected is quite small compared to the total number of rows, I will go ahead and remove the affected rows. """

# To delete the affected rows
affected_rows = income_data[income_data['workclass'] == ' ?'].index
income_data.drop(affected_rows, inplace=True)

# For education
print(income_data['education'].unique())

"""The eductation column has the right values."""

# For marital-status
print(income_data['marital_status'].unique())

"""The marital status columns also looks good."""

# For occupation
print(income_data['occupation'].unique())

"""I can still see the '?' value in this column, I will also retrieve the affected rows


"""

# To retrieve affected rows
income_data[income_data['occupation'] == ' ?']

"""Only a few rows are affected, they will be dropped."""

# To delete the affected row
affected_rows1 = income_data[income_data['occupation'] == ' ?'].index
income_data.drop(affected_rows1, inplace=True)

# For relationship
print(income_data['relationship'].unique())

"""This column aslo looks good."""

# For race
print(income_data['race'].unique())

"""The race column looks good"""

# For sex
print(income_data['sex'].unique())

"""The sex column looks good"""

# For native country
print(income_data['native_country'].unique())

# To delete the affected rows
affected_rows2 = income_data[income_data['native_country'] == ' ?'].index
income_data.drop(affected_rows2, inplace=True)

# To check the size of the data after the deletion
income_data_shape = income_data.shape
print('This dataset now has {} rows and {} columns'.format(income_data_shape[0], income_data_shape[1]))

"""### BAYESIAN NETWORK"""

income_data.columns

# To create a network
model = BayesianNetwork([('age', 'workclass'), ('workclass','income'), ('age', 'education'), ('education','income'),
                      ('age', 'income'), ('age', 'occupation'), ('sex','occupation'), ('occupation','hours_per_week'),
                      ('occupation','income'), ('occupation', 'capital_loss'), ('occupation', 'capital_gain'),    
                      ('hours_per_week', 'income')])

# To Fit the model
model.fit(income_data, estimator=MaximumLikelihoodEstimator)


# To make inferences using the model
infer = VariableElimination(model)

"""query. What is the probabilty of a person's occupation given that the person is female?"""

query1=infer.query(variables=['occupation'],evidence={'sex':' Female'})
print(query1)


# To visualize the Directed Acyclic Graph
# To create a function that can be used to resize the graph
def __set_size(width : int ,height : int):
    """Explicitly sets the size of a matplotlib plot
    Args:
        width (int): Width in inches
        height (int): Height in inches
    References:
        https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    """
    axis=plt.gca()

    figw = float(width)/(axis.figure.subplotpars.right - axis.figure.subplotpars.left)
    figh = float(height)/(axis.figure.subplotpars.top-axis.figure.subplotpars.bottom)

    axis.figure.set_size_inches(figw, figh)
    
# plotting the graph    
model.to_daft().render()
__set_size(6, 4)
plt.show()

