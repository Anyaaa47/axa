# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:39:15 2020

@author: DBrughmans
"""
from scipy import stats
import pandas as pd
import numpy as np

def Preprocessing_discrete(Var, train_ind, encoding_opt, Var_name, thermo_categories=[''], thermo_opt="normal"):
    """
    Args:
        Var: represents the single variable you wish to preprocess,
        should be a numerical variable. Make sure the data type is "Pandas.Series".

        train_ind: the positions (indices) of the training data

        encoding_opt: 'dummy' or 'thermometer'

        thermo_categories: represents the unique categories in ascending order
        eg, ['A1', 'A2', 'A3']
        Important: dtype=list

        thermo_opt; 'normal' or 'special'
        'normal' applies a standard thermometer encoding
        'special' encodes last category by dummy encoding

        Var_name: name of variable

    Function replaces missing values by the mode of the training data
    and subsequently applies a dummy or thermometer encoding

    Returns the preprocessed variable as well as the categories
    that each column corresponds with.
    """
    Var_train=Var.iloc[train_ind]
    Var_pre = Var.fillna(value = Var_train.mode().iloc[0])#Replace mising values of feature with the mode of the training set

    #Dummy encoding or thermometer encoding
    if encoding_opt=='dummy':
        Var_preprocessed=pd.get_dummies(Var_pre,prefix= str(Var_name))
        Categories=list(Var_preprocessed.columns.values) #show categories

        # from a categorical variable with k categories, we only create k-1 dummy variables
        Reference_Category = Categories[0]  # the dropped category becomes the reference category
        Var_preprocessed = Var_preprocessed.iloc[:, 1:]
        Categories = Categories[1:]

    elif (encoding_opt=='thermometer'):
        for j, value in enumerate(thermo_categories):
            Var_pre = Var_pre.replace(value,j)
            max_value = j
        Categories = [Var_name + '_'+category for category in thermo_categories]
        Reference_Category = Categories[0]
        Categories = Categories[1:]
        Var_preprocessed = pd.DataFrame(
            data = (np.arange(max_value) < np.array(Var_pre).reshape(-1, 1)).astype(int),
            columns = Categories,
            index = Var_pre.index
        )
        if thermo_opt=='special':
            special_encoding = np.zeros(Var_preprocessed.shape[1],dtype = int)
            special_encoding[-1]= 1
            Var_preprocessed.loc[Var_preprocessed.iloc[:,-1]==1,:]=special_encoding


    return (Var_preprocessed, Categories, Reference_Category)
        