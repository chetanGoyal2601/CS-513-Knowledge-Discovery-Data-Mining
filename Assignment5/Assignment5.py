#Chetan Goyal
#importing the required modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#reading the breast cancer file into a datframe
df = pd.read_csv("breast-cancer-wisconsin.csv")


#removing missing values
df = df[df.F6 != '?']


#creating features for test and train
X = df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']]
y = df[['Class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)

#creating our Decision Tree Classifier - CART classifier is what is packaged into sklearn
clf = DecisionTreeClassifier().fit(X_train, np.ravel(y_train,order='C'))

#Estimating the accuracy
print(clf.score(X_test, y_test))
# Accuracy -> 0.9414634146341463



#Making a confusion matrix
df_confusion_matrix = df.copy()
df_confusion_matrix = df_confusion_matrix.reset_index()
df_confusion_matrix['F6'] = df_confusion_matrix['F6'].astype('int64')

for i, row in df_confusion_matrix.iterrows():
    df_confusion_matrix.at[i, 'Predicted Class'] = clf.predict([[row['F1'], row['F2'], row['F3'], row['F4'], row['F5'], row['F6'], row['F7'], row['F8'], row['F9']]])[0]
    
    
df_confusion_matrix['Predicted Class'] =  df_confusion_matrix['Predicted Class'].astype(int)
#print(df_confusion_matrix)
df_confusion_matrix.to_csv("Confusion_Matrix.csv")






