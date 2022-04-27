import uuid

import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import valohai

def main():
  
  valohai.prepare(
        step='train-model',
        image='valohai/sklearn:1.0',
        default_inputs={
          'dataset1': 'datum://01806aca-ad00-8e7c-5476-1e24634f85aa',
          'dataset2': 'datum://01806aca-b05d-67a8-035d-194c86a329bb',
          'dataset3': 'datum://01806aca-aecb-98bf-daf0-733753a962a6',
          'dataset4': 'datum://01806aca-b1f6-35bc-08a7-0199ffeedbf5',
        },
    )
  
  x_train_dup = valohai.inputs('dataset1').path()
  y_train_dup = valohai.inputs('dataset2').path()
  x_test_dup = valohai.inputs('dataset3').path()
  y_test_dup = valohai.inputs('dataset4').path()
  
  x_train = pd.read_csv(x_train_dup,index_col=0)
  y_train = pd.read_csv(y_train_dup,index_col=0)
  x_test = pd.read_csv(x_test_dup,index_col=0)
  y_test = pd.read_csv(y_test_dup,index_col=0)
  
  rf = RandomForestClassifier(n_estimators=100, random_state=1,verbose=0)
  rf.fit(x_train, y_train)
  y_pred = rf.predict(x_test)
  print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
  test_accuracy_rf = accuracy_score(y_test, y_pred)
  test_accuracy_rf_score=rf.score(x_test,y_test)*100
  
  with valohai.logger() as logger:
      logger.log('test_accuracy_rf', test_accuracy_rf)
      logger.log('test_accuracy_rf_score', test_accuracy_rf_score)
      
  print("The confusion matrix")
  cm = confusion_matrix(y_test, y_pred)
  plt.rcParams['figure.figsize'] = (5, 5)
  sns.set(style = 'dark', font_scale = 1.4)
  sns.heatmap(cm, annot = True, annot_kws = {"size": 15})
 
  suffix = uuid.uuid4()
  save_path = valohai.outputs().path('confusion_matrix.png')
  plt.savefig(save_path)
  plt.show()
  plt.close()
  output_path = valohai.outputs().path('model_rf.pckl')
  joblib.dump(rf, open(output_path, 'wb'))
  #rf.save(output_path)

if __name__ == '__main__':
    main()
