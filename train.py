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
          'dataset1': 'datum://018069d3-4b89-f0cf-f815-c51e53838660',
          'dataset2': 'datum://018069d3-4eaa-5d5a-1f7e-60812c5add84',
          'dataset3': 'datum://018069d3-4d1c-1722-a671-703994d1ea2c',
          'dataset4': 'datum://018069d3-5044-638c-ba9b-16dceb277d26',
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
