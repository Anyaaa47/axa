# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:31:37 2021

@author: 79217
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel(r'D:\challenge\work\DSC_2021_Training_01labels.xlsx')

Data_preprocessed=[]

from sklearn.model_selection import train_test_split
indices=np.arange(1000)
indices_train, indices_test  = train_test_split(indices, test_size=0.2, random_state=0)
indices_train, indices_val  = train_test_split(indices_train, test_size=0.2, random_state=0)

df= df.drop('FREE_CASH_FLOW_AMT', axis=1)
df= df.drop('MTHS_FIRST_PCX_COREPRIV_CNT', axis=1)
df= df.drop('MONTHS_SINCE_LAST_REFUSAL_CNT', axis=1)
df= df.drop('A2_MTHS_SNC_FIRST_COREPROF_CNT', axis=1)
df= df.drop('LoanID', axis=1)

#df.isnull().sum() 
#missing_count = df.isnull().sum() # the count of missing values 
#value_count = df.isnull().count() # the count of all values 
#missing_percentage = round(missing_count/value_count*100,1)
#missing_df = pd.DataFrame({'count': missing_count, 'percentage': missing_percentage}) #create a dataframe 
#print(missing_df)

Data=df.iloc[:,0:45]

Beh = Data['BEH_SCORE_AVG_A1']
Type = Data['Type']
Label = Data['Label_Default']

from function1.Preprocessing_continuous import Preprocessing_continuous

Beh_preprocessed = Preprocessing_continuous(Beh, indices_train)
Data_preprocessed.append(Beh_preprocessed)

from function2.Preprocessing_discrete import Preprocessing_discrete


Type_preprocessed = Preprocessing_discrete(Type, indices_train, 'dummy', 'Type')
Data_preprocessed.append(Type_preprocessed[0])


Data_preprocessed_full= pd.concat(Data_preprocessed, axis=1)

with pd.ExcelWriter('Preprocessed_data.xlsx') as writer:
    Data_preprocessed_full.to_excel(writer, sheet_name='Data preprocessed')
    Label.to_excel(writer, sheet_name='Label_Default')
    
    pd.DataFrame(indices_test).to_excel(writer, sheet_name='indices test')
    pd.DataFrame(indices_train).to_excel(writer, sheet_name='indices train')
    pd.DataFrame(indices_val).to_excel(writer, sheet_name='indices val')