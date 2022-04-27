import uuid

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import valohai

def main():
  
  valohai.prepare(
        step='train-model',
        image='valohai/sklearn:1.0',
        default_inputs={
          'dataset1': 'datum://01806971-5724-09f4-2604-4ca1dda335b6',
          'dataset2': 'datum://01806971-58b4-8257-dffe-3a2b02aef9c7',
          'dataset3': 'datum://01806971-5a3f-726d-03da-15920ab9cade',
          'dataset4': 'datum://01806971-5bca-2dda-158b-15d70de54d9f',
        },
    )
  
  x_train = pd.read_csv(valohai.inputs('dataset1').path())
  y_train = pd.read_csv(valohai.inputs('dataset2').path())
  x_test = pd.read_csv(valohai.inputs('dataset3').path())
  y_test = pd.read_csv(valohai.inputs('dataset4').path())
  
  rf = RandomForestClassifier(n_estimators=100, random_state=1,verbose=0)
  rf.fit(x_train, y_train)
  y_pred2 = rf.predict(x_test)
  test_accuracy_rf = rf.score(x_test,y_test)*100
  
  with valohai.logger() as logger:
      logger.log('test_accuracy_rf', test_accuracy_rf)
      
  # printing the confusion matrix
  #cm = confusion_matrix(y_test_os, y_pred)
  #sns.heatmap(cm, annot = True, cmap = 'rainbow')
  #print("Accuracy: ", lr.score(x_test_os,y_test_os)*100)
  #cm = confusion_matrix(y_test_os, y_pred)
  #sns.heatmap(cm, annot = True, cmap = 'rainbow')
  suffix = uuid.uuid4()
  output_path = valohai.outputs().path('model_rf.h5')
  rf.save(output_path)

if __name__ == '__main__':
    main()
