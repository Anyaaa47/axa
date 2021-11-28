import numpy as np

def Preprocessing_continuous(Var, train_ind):
    
    """
    Args:
        Var: represents the single variable you wish to preprocess,
        should be a numerical variable. Make sure the data type is "Series".
        train_ind: the positions (indices) of the training data
    """
    Var_train=Var.iloc[train_ind]
    Var_train_mean = np.mean(Var_train) 
    Var_train_std = np.std(Var_train) 
    
    #Replace mising values of feature with the mean of the training set
    Var.fillna(value=Var_train_mean, inplace=True)
    
    #Standardize variable
    Var_preprocessed = (Var-Var_train_mean)/Var_train_std
    
    #Deal with outliers
    j=0 #start with first instance
    for i in Var_preprocessed:
        if (i>3): 
            Var_preprocessed.iloc[j] = 3
        elif (i<-3):
            Var_preprocessed.iloc[j] = -3
        j+=1 
    
    return(Var_preprocessed)

    

