import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import valohai


def main():
    # valohai.prepare enables us to update the valohai.yaml configuration file with
    # the Valohai command-line client by running `valohai yaml step preprocess_dataset.py`

    valohai.prepare(
        step='Preprocess Dataset',
        image='python:3.9',
        default_inputs={
            'dataset': 'https://depprocureformstorage.blob.core.windows.net/semicond-yield/input/uci-secom.csv?sp=r&st=2022-04-18T07:59:29Z&se=2022-04-18T15:59:29Z&spr=https&sv=2020-08-04&sr=b&sig=1nstM7cXFjUFZbjtiQTV7HODf2NEQnmie2chSy4rjCE%3D'
        },
    )

    # Read input files from Valohai inputs directory
    # This enables Valohai to version your training data
    # and cache the data for quick experimentation

    print('Loading data')
    data = pd.read_csv(valohai.inputs('dataset').path())
    data = data.drop(columns = ['Time'], axis = 1)
    check = data.isnull().any().any()    
    if check:
       data = data.replace(np.NaN, 0)
    X = data.drop(columns=['Pass/Fail'],axis=1)
    y = data["Pass/Fail"]
    
    print('Feature Selection using LassoCV started')
    model = LassoCV(max_iter=10000)
    model.fit(X,y)
    print(model.get_params(deep=True))
    print("Best Alpha using built-in LassoCV is: %f" % model.alpha_)
    print("Best score using built-in LassoCV is: %f" % model.score(X,y))
    coef=pd.Series(model.coef_,index=X.columns)

    print("Lasso picked "+ str(sum(coef!= 0))+ " features and eliminated the other "+ str(sum(coef == 0))+" variables")
    
    column=[]
    selected_columns=[]
    for i in range(len(coef)):
        if coef[i]!=0:
            column.append(i)
    print("Selected columns from LassoCV is: ",column)
    selected_columns = [str(x) for x in column]
    X = pd.DataFrame(X, columns=selected_columns)
    
    print("Preparing for Undersampling")
    lasso_data=pd.concat([X,y],axis=1)
    failed_tests = np.array(lasso_data[lasso_data['Pass/Fail'] == 1].index)
    no_failed_tests = len(failed_tests)
    print("The number of failed tests(1) in data:",no_failed_tests)
    
    normal_indices = lasso_data[lasso_data['Pass/Fail'] == -1]
    no_normal_indices = len(normal_indices)
    print("The number of passed tests(-1) in data:",no_normal_indices)
    
    random_normal_indices = np.random.choice(no_normal_indices, size = no_failed_tests, replace = True)
    random_normal_indices = np.array(random_normal_indices)
    print("The number of randomly choosen passed tests(-1) in data:",len(random_normal_indices))

    under_sample = np.concatenate([failed_tests, random_normal_indices])
    print("The length of under sampled data:",len(under_sample))
    
    undersample_data = lasso_data.iloc[under_sample, :]
    x_us = undersample_data.iloc[:, undersample_data.columns != 'Pass/Fail'] 
    y_us = undersample_data.iloc[:, undersample_data.columns == 'Pass/Fail']

    x_train, x_test, y_train, y_test = train_test_split(x_us, y_us, test_size = 0.2, random_state = 1)
    print(y_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test=pd.DataFrame(y_test)
  
    
    print('Saving preprocessed data')
    path_x_train = valohai.outputs("preprocessed_yield_uci").path('x_train.csv')
    x_train.to_csv(path_x_train)
    path_x_test = valohai.outputs("preprocessed_yield_uci").path('x_test.csv')
    x_test.to_csv(path_x_test)
    path_y_train = valohai.outputs("preprocessed_yield_uci").path('y_train.csv')
    y_train.to_csv(path_y_train)
    path_y_test = valohai.outputs("preprocessed_yield_uci").path('y_test.csv')
    y_test.to_csv(path_y_test)
    
if __name__ == '__main__':
    main()
