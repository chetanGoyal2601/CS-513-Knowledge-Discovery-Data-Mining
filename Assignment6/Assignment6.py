#Chetan Goyal
#importing the required modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#reading the breast cancer file into a datframe
df = pd.read_csv("breast-cancer-wisconsin.csv")

#removing missing values
df = df[df.F6 != '?']

#creating features for test and train
X = df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']]
y = df[['Class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)

#6.1 - C5.0 Methodology through DecisionTreeClassfiier Package (CART)
from sklearn.tree import DecisionTreeClassifier
clf_C5 = DecisionTreeClassifier().fit(X_train, np.ravel(y_train,order='C'))

#Estimating the accuracy
print(clf_C5.score(X_test, y_test))
# Accuracy -> 0.9414634146341463

#Making a confusion matrix for C5
df_confusion_matrix_C5 = df.copy()
df_confusion_matrix_C5 = df_confusion_matrix_C5.reset_index()
df_confusion_matrix_C5['F6'] = df_confusion_matrix_C5['F6'].astype('int64')

for i, row in df_confusion_matrix_C5.iterrows():
    df_confusion_matrix_C5.at[i, 'Predicted Class'] = clf_C5.predict([[row['F1'], row['F2'], row['F3'], row['F4'], row['F5'], row['F6'], row['F7'], row['F8'], row['F9']]])[0]
    
df_confusion_matrix_C5['Predicted Class'] =  df_confusion_matrix_C5['Predicted Class'].astype(int)
#print(df_confusion_matrix_C5)
df_confusion_matrix_C5.to_csv("Confusion_Matrix_C5.csv")



#6.2 - Creating a classifier using Random Forest
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier().fit(X_train, np.ravel(y_train,order='C'))

#Estimating the accuracy
print(clf_RF.score(X_test, y_test))
# Accuracy -> 0.9463414634146341

#Making a confusion matrix for C5
df_confusion_matrix_RF = df.copy()
df_confusion_matrix_RF = df_confusion_matrix_RF.reset_index()
df_confusion_matrix_RF['F6'] = df_confusion_matrix_RF['F6'].astype('int64')

for i, row in df_confusion_matrix_RF.iterrows():
    df_confusion_matrix_RF.at[i, 'Predicted Class'] = clf_RF.predict([[row['F1'], row['F2'], row['F3'], row['F4'], row['F5'], row['F6'], row['F7'], row['F8'], row['F9']]])[0]
    
df_confusion_matrix_RF['Predicted Class'] =  df_confusion_matrix_RF['Predicted Class'].astype(int)
#print(df_confusion_matrix_RF)
df_confusion_matrix_RF.to_csv("Confusion_Matrix_RF.csv")
