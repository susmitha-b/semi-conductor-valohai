from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import valohai

def main():
  
  valohai.prepare(
        step='train-model',
        image='valohai/sklearn:1.0',
        default_inputs={
            'dataset1': 'datum://01803c30-a8a1-4530-bda1-6eee87c9a3fa',
        },
        default_parameters={
            'learning_rate': 0.001,
            'epochs': 5,
        },
    )
  
  x_resample, y_resample  = SMOTE(random_state=1).fit_sample(x, y.values.ravel())

#print(x_resample.shape)
#print(y_resample.shape)

x_train_os, x_test_os, y_train_os, y_test_os = train_test_split(x, y, test_size = 0.3, random_state = 1)

print(x_train_os.shape)
print(y_train_os.shape)
print(x_test_os.shape)
print(y_test_os.shape)

from sklearn.preprocessing import StandardScaler

# creating a standard scaler
sc = StandardScaler()

sc = StandardScaler()
x_train_os = sc.fit_transform(x_train_os)
x_test_os = sc.transform(x_test_os)

model = RandomForestClassifier(n_estimators=100, random_state=1,verbose=0 )
model.fit(x_train_os, y_train_os)
#scores_prediction = model.decision_function(x_train)
y_pred = model.predict(x_test_os)

print("Accuracy: ", model.score(x_test_os,y_test_os)*100)

# printing the confusion matrix
cm = confusion_matrix(y_test_os, y_pred)
sns.heatmap(cm, annot = True, cmap = 'rainbow')

lr = LogisticRegression(random_state=1)
lr.fit(x_train_os, y_train_os) 
y_pred = lr.predict(x_test_os)

print("Accuracy: ", lr.score(x_test_os,y_test_os)*100)

cm = confusion_matrix(y_test_os, y_pred)
sns.heatmap(cm, annot = True, cmap = 'rainbow')
