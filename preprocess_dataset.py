import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import valohai
from imblearn.over_sampling import SMOTE


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
    check = data.isnull().any().any()
    if check:
       data = data.replace(np.NaN, 0)
    def remove_collinear_features(x, threshold):
        corr_matrix = x.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the threshold
                if val >= threshold:
                # Print the correlated features and the correlation value
                   #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                   drop_cols.append(col.values[0])

            # Drop one of each pair of correlated columns
        drops = set(drop_cols)
        x = x.drop(columns=drops)

        return x       
            
    data = remove_collinear_features(data,0.70)
        
    # separating the dependent and independent data
    x = data.iloc[:,:306]
    y = data["Pass/Fail"]
        
    print('Preprocessing data')
    x_resample, y_resample  = SMOTE(random_state=1).fit_sample(x, y.values.ravel())
    #x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size = 0.3, random_state = 1)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    # Write output files to Valohai outputs directory
    # This enables Valohai to version your data
    # and upload output it to the default data store

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
