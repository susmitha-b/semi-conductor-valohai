import uuid

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import valohai

def main():
  
  valohai.prepare(
        step='train-model',
        image='valohai/sklearn:1.0',
        default_inputs={
          'dataset1': 'datum://01803ca5-4e35-f0d4-beb6-4b7031282885',
          'dataset2': 'datum://01803ca5-51fe-665d-80fb-483245693846',
          'dataset3': 'datum://01803ca5-505f-996f-c4c6-0f9c91118a09',
          'dataset4': 'datum://01803ca5-538f-8997-9202-597921739ea8',
        },
        default_parameters={
          'random_state': 1,
          'max_depth': 4,
          'n_jobs':4,
          'cv':2,
          'n_estimators':100,
        },
    )
  
  x_train = pd.read_csv(valohai.inputs('dataset1').path())
  y_train = pd.read_csv(valohai.inputs('dataset2').path())
  x_test = pd.read_csv(valohai.inputs('dataset3').path())
  y_test = pd.read_csv(valohai.inputs('dataset4').path())
  
  y_train_repl=y_train.replace(-1,0)
  y_test_repl=y_test.replace(-1,0)
  clf = xgb.XGBClassifier(max_depth=6,n_estimators=100)
  clf.fit(x_train, y_train_repl)
  y_pred1 = clf.predict(x_test)
  test_accuracy_xgb = clf.score(x_test,y_test_repl)*100
  
  rf = RandomForestClassifier(n_estimators=valohai.parameters('n_estimators').value, random_state=valohai.parameters('random_state').value,verbose=0)
  rf.fit(x_train, y_train)
  y_pred2 = rf.predict(x_test)
  test_accuracy_rf = rf.score(x_test,y_test)*100
  
  lr = LogisticRegression(random_state=1)
  lr.fit(x_train, y_train) 
  y_pred3 = lr.predict(x_test)
  test_accuracy_lr = lr.score(x_test,y_test)*100
  
  with valohai.logger() as logger:
      logger.log('test_accuracy_xgb', test_accuracy_xgb)
      logger.log('test_accuracy_rf', test_accuracy_rf)
      logger.log('test_accuracy_lr', test_accuracy_lr)
      
  # printing the confusion matrix
  #cm = confusion_matrix(y_test_os, y_pred)
  #sns.heatmap(cm, annot = True, cmap = 'rainbow')
  #print("Accuracy: ", lr.score(x_test_os,y_test_os)*100)
  #cm = confusion_matrix(y_test_os, y_pred)
  #sns.heatmap(cm, annot = True, cmap = 'rainbow')
  suffix = uuid.uuid4()
  output_path1 = valohai.outputs().path('model_xgb.h5')
  output_path2 = valohai.outputs().path('model_rf.h5')
  output_path3 = valohai.outputs().path('model_lr.h5')
  xgb.save(output_path1)
  rf.save(output_path2)
  lr.save(output_path3)


if __name__ == '__main__':
    main()
