# -*- coding: utf-8 -*-
"""Feature_Preprocessing_Data_Leakage.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17MfwfZWRI2IHR9qvfJJS7Im-0Xwc4daw

# Feature preprocessing and data leakage

This small tutorial aims to show the importance of correct data preprocessing to avoid data leakage.

A (very good) full article about this problem can be found [here](https://machinelearningmastery.com/data-preparation-without-data-leakage/).

**Question:** Should I use entire dataset (train/validation/test) in preprocessing steps or just the training set?

**Short answer**: You may apply preprocessing in the train set and, then, transform the test/validation set.

# Step 0 - Preparing the environment
"""

import urllib.request

# First let's download the datasets
#urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")
#urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "adult.test")

"""# Step 1 - Loading the dataset"""

import pandas as pd

# Now, load the train and test dataset into Pandas Dataframes

df_train = pd.read_excel(r'D:\challenge\work\DSC_2021_Training_001labels.xlsx', header=None)
df_test = pd.read_excel(r'D:\challenge\work\DSC_2021_0Test.xlsx', header=None)

#df_train= df_train.drop('FREE_CASH_FLOW_AMT', axis=1)
#df_train= df_train.drop('MTHS_FIRST_PCX_COREPRIV_CNT', axis=1)
#df_train= df_train.drop('MONTHS_SINCE_LAST_REFUSAL_CNT', axis=1)
#df_train= df_train.drop('A2_MTHS_SNC_FIRST_COREPROF_CNT', axis=1)
#df_train= df_train.drop('LoanID', axis=1)



#df_test= df_test.drop('FREE_CASH_FLOW_AMT', axis=1)
#df_test= df_test.drop('MTHS_FIRST_PCX_COREPRIV_CNT', axis=1)
#df_test= df_test.drop('MONTHS_SINCE_LAST_REFUSAL_CNT', axis=1)
#df_test= df_test.drop('A2_MTHS_SNC_FIRST_COREPROF_CNT', axis=1)
#df_test= df_test.drop('LoanID', axis=1)

#df_train= df_train.drop('Product_Desc', axis=1)
#df_test= df_test.drop('Product_Desc', axis=1)

# Let's include, manually, the columns names (https://archive.ics.uci.edu/ml/machine-learning-databases/adult/old.adult.names)
column_names = ['LoanID','Type','INDUSTRY_CD_3','INDUSTRY_CD_4','BEH_SCORE_AVG_A1','BEH_SCORE_AVG_A2','BEH_SCORE_MIN_A2','FHS_SCORE_AVG','FHS_SCORE_LATEST','CASH_FLOW_AMT','FREE_CASH_FLOW_AMT','A2_TOTAL_EMPLOYMENT_MONTHS_CNT','MTHS_SNC_1ST_REC_CNT','Managing_Sales_Office_Nbr','Postal_Code_L','A1_AVG_POS_SALDO_PROF_1_AMT','Invest_Amt','Original_loan_Amt','MTHS_FIRST_PCX_COREPRIV_CNT','CREDIT_TYPE_CD','ACCOUNT_PURPOSE_CD','A2_AVG_NEG_SALDO_PRIV_12_AMT','A2_ANNUAL_INCOME_AMT','A1_TOT_DEB_INTEREST_PROF_6_AMT','A2_MTHS_SNC_LAST_LIQ_PRIV_CNT','A2_MTHS_SNC_FIRST_COREPROF_CNT','A2_MARITAL_STATUS_CD','A1_TOT_DEB_INTEREST_PROF_1_AMT','MTHS_IN_BUSINESS_CNT','A2_MTHS_SNC_LAST_LIQ_SAVE_CNT','A1_NEGAT_TRANS_COREPROF_CNT','FINANCIAL_PRODUCT_TYPE_CD','A1_OVERDRAWN_DAYS_PROF_24_CNT','A1_OVERDRAWN_DAYS_PROF_6_CNT','A2_EMPLOYMENT_STATUS_CD','A2_AVG_POS_SALDO_SAVINGS_12_AMT','A1_AVG_NEG_SALDO_PROF_3_AMT','A1_AVG_POS_SALDO_PROF_12_AMT','A2_RESIDENT_STATUS_CD','A1_TOT_STND_PAYMENT_INT_PROF_CNT','CASHFLOW_MONTHLY_CREDIT_RT','MONTHS_SINCE_LAST_REFUSAL_CNT','MONTHLY_CREDIT_AMT_TOT', 'Label_Default']

df_train.columns = column_names
df_test.columns = column_names



# Then, let's see how the dataframe looks like
df_train.head(5)

"""# Step 2 - Dataset preprocessing

Now, we will **fit** different preprocessing methodologies in the **TRAINING** (`df_train`) set and then **transform** both the **TRAINING** (`df_train`) and **TEST** (`df_test`) sets.

We will use:

For continuous features:
* Statistical normalization using [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

For categorical features:
* Weight of Evicence using [WOEEncoder](https://contrib.scikit-learn.org/category_encoders/woe.html)
* One-hot encoding using [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
* Target encoding using [TargetEncoder](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
"""

# Now, let's make a list with the features we are going to preprocess

# Features that we will apply the StandardScaler
feat_cont_statnorm = ['MTHS_FIRST_PCX_COREPRIV_CNT','FREE_CASH_FLOW_AMT','MONTHS_SINCE_LAST_REFUSAL_CNT','A2_MTHS_SNC_FIRST_COREPROF_CNT','BEH_SCORE_AVG_A1','BEH_SCORE_AVG_A2','BEH_SCORE_MIN_A2','FHS_SCORE_AVG','FHS_SCORE_LATEST','CASH_FLOW_AMT','A2_TOTAL_EMPLOYMENT_MONTHS_CNT','MTHS_SNC_1ST_REC_CNT', 'A1_AVG_POS_SALDO_PROF_1_AMT','Invest_Amt','Original_loan_Amt','A2_AVG_NEG_SALDO_PRIV_12_AMT','A2_ANNUAL_INCOME_AMT','A1_TOT_DEB_INTEREST_PROF_6_AMT','A2_MTHS_SNC_LAST_LIQ_PRIV_CNT','A1_TOT_DEB_INTEREST_PROF_1_AMT','MTHS_IN_BUSINESS_CNT','A2_MTHS_SNC_LAST_LIQ_SAVE_CNT','A1_NEGAT_TRANS_COREPROF_CNT','A1_OVERDRAWN_DAYS_PROF_24_CNT','A1_OVERDRAWN_DAYS_PROF_6_CNT','A2_AVG_POS_SALDO_SAVINGS_12_AMT','A1_AVG_NEG_SALDO_PROF_3_AMT','A1_AVG_POS_SALDO_PROF_12_AMT','A1_TOT_STND_PAYMENT_INT_PROF_CNT','CASHFLOW_MONTHLY_CREDIT_RT','MONTHLY_CREDIT_AMT_TOT']

# Features that we will apply WOEEncoder
feat_cat_woe = ['LoanID','INDUSTRY_CD_3','INDUSTRY_CD_4','A2_EMPLOYMENT_STATUS_CD','ACCOUNT_PURPOSE_CD','Managing_Sales_Office_Nbr','Postal_Code_L']

# Features that we will apply OneHotEncoder
feat_cat_ohe = ['Type','CREDIT_TYPE_CD','FINANCIAL_PRODUCT_TYPE_CD','A2_RESIDENT_STATUS_CD']

# Features that we will apply TargetEncoder
feat_cat_te = ['A2_MARITAL_STATUS_CD']

"""## 2.1 - Continuous feature statistical normalization"""

# First, let's work with continuous features
# Here we will verify some rows of the columns we will modify

# For train:
df_train[feat_cont_statnorm].head(5)

# For test:
df_test[feat_cont_statnorm].head(5)

# To normalize, first we will import the normalization package
from sklearn.preprocessing import StandardScaler

# Then we will create a scaler to use in our preprocessing
scaler_statistical_norm = StandardScaler()

# Now, we will fit the scaler with OUR TRAIN DATA (using the specific features  
# indicated in feat_cont_statnorm), this creates a function
# that can transform raw data to normalized data
scaler_statistical_norm.fit(df_train[feat_cont_statnorm])

# Following, we will transform (using scaler_statistical_norm.transform)
# our train data using the scaler
df_train[feat_cont_statnorm] = scaler_statistical_norm.transform(df_train[feat_cont_statnorm])

# And we will do the same for the test set, notice WE DIDN'T FIT THE TEST SET
# we only fit the scaler in the training and then use it for train and test
df_test[feat_cont_statnorm] = scaler_statistical_norm.transform(df_test[feat_cont_statnorm])

# Now, let's see the our modifications!
# For train:
df_train[feat_cont_statnorm].head(5)

# For test:
df_test[feat_cont_statnorm].head(5)

"""## 2.2 - Categorical features weight of evidence"""

# Now, let's do a similar thing, but for categorical datasets
# First we will use WoE approach

# Then, let's see the train dataset we will modify:
df_train[feat_cat_woe].head(5)

# Now for test
df_test[feat_cat_woe].head(5)

# To encode, first we will import the encoding package
from category_encoders.woe import WOEEncoder

# Then we will create a encoder to use in our preprocessing
encoder_woe = WOEEncoder()

# Now, we will fit the encoder with OUR TRAIN DATA (using the specific features  
# indicated in feat_cat_woe), this creates a function
# that can transform raw data to encoded data
# For this encoder, we need the training label and we specify it by df_train['label']
encoder_woe.fit(df_train[feat_cat_woe], df_train['Label_Default'])

# Following, we will encode (using encoder_woe.transform)
# our train data using the encoder
df_train[feat_cat_woe] = encoder_woe.transform(df_train[feat_cat_woe])

# And we will do the same for the test set, notice WE DIDN'T FIT THE TEST SET
# we only fit the encoder in the TRAINING and then use it to transform the train and test sets
df_test[feat_cat_woe] = encoder_woe.transform(df_test[feat_cat_woe])

# Now, let's see how the train features were modified:
df_train[feat_cat_woe].head(5)

# For the test set:
df_test[feat_cat_woe].head(5)

"""## 2.3 - Categorical features one-hot encoding"""

# Now we will use one-hot encode approach

# Then, let's see the train dataset we will modify:
df_train[feat_cat_ohe].head(5)

# Now for test
df_test[feat_cat_ohe].head(5)

# To encode, first we will import the encoding package
from sklearn.preprocessing import OneHotEncoder

# Then we will create a encoder to use in our preprocessing
encoder_ohe = OneHotEncoder()

# Now, we will fit the encoder with OUR TRAIN DATA (using the specific features  
# indicated in feat_cat_ohe), this creates a function
# that can transform raw data to encoded data
encoder_ohe.fit(df_train[feat_cat_ohe])

# Then, since the one-hot encoding creates new columns, we will get
# those columns first
new_ohe_columns_train = encoder_ohe.transform(df_train[feat_cat_ohe])

# And we will do the same for the test set, notice WE DIDN'T FIT THE TEST SET
# we only fit the encoder in the TRAINING and then use it to create the test set columns
new_ohe_columns_test = encoder_ohe.transform(df_test[feat_cat_ohe])




# Now, since the new columns are sparse matrixes, we need to conver them to
# pandas DataFrames, let's do for train:
df_new_ohe_columns_train = pd.DataFrame(new_ohe_columns_train.toarray())

# Then, for test:
df_new_ohe_columns_test = pd.DataFrame(new_ohe_columns_test.toarray())



# Next, let's fix the labels names, since now the labels are numerical
# this can be done getting the labels name by encoder_ohe.get_feature_names_out()
# For train:
df_new_ohe_columns_train.columns = encoder_ohe.get_feature_names_out()
# For test:
df_new_ohe_columns_test.columns = encoder_ohe.get_feature_names_out()



# Then, we now have to drop the original columns from the dataframe, and then
# replace them with the new, encoded, features

# For train:
# * Dropping the original features that we encoded
df_train = df_train.drop(columns=feat_cat_ohe)
# * Adding the new OHE features
df_train = df_train.join(df_new_ohe_columns_train)

# For test:
# * Dropping the original features that we encoded
df_test = df_test.drop(columns=feat_cat_ohe)
# * Adding the new OHE features
df_test = df_test.join(df_new_ohe_columns_test)

# Now, let's see the new columns (names in encoder_ohe.get_feature_names_out()) 
# for train:
df_train[encoder_ohe.get_feature_names_out()].head(5)

# For test:
df_test[encoder_ohe.get_feature_names_out()].head(5)

"""## 2.4 - Categorical features target encode"""

# Now we will use target encode approach

# Then, let's see the train dataset we will modify:
df_train[feat_cat_te].head(5)

# Now for test
df_test[feat_cat_te].head(5)

# To encode, first we will import the encoding package
from category_encoders.target_encoder import TargetEncoder

# Then we will create a encoder to use in our preprocessing
encoder_te = TargetEncoder()

# Now, we will fit the encoder with OUR TRAIN DATA (using the specific features  
# indicated in feat_cat_te), this creates a function
# that can transform raw data to encoded data
# For this encoder, we need the training label and we specify it by df_train['label']
encoder_te.fit(df_train[feat_cat_te], df_train['Label_Default'])

# Following, we will encode (using encoder_te.transform)
# our train data using the encoder
df_train[feat_cat_te] = encoder_te.transform(df_train[feat_cat_te])

# And we will do the same for the test set, notice WE DIDN'T FIT THE TEST SET
# we only fit the encoder in the TRAINING and then use it to transform the train and test sets
df_test[feat_cat_te] = encoder_te.transform(df_test[feat_cat_te])

# Now, let's see how the train features were modified:
df_train[feat_cat_te].head(5)

# For the test set:
df_test[feat_cat_te].head(5)

"""# Step 3 - Output datasets"""

# Now, we have the final datasets for train:
df_train.head(5)

# And for test:
df_test.head(5)

"""Those datasets are normalized and encoded, ready to be used for training (the train set, `df_train`) and  testing (the test set, `df_test`)"""

a = 1
