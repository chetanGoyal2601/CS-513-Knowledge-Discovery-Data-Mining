#importing the required modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#reading the breast cancer file into a datframe
df = pd.read_csv("breast-cancer-wisconsin.csv")
print(df)


#removing missing values
df = df[df.F6 != '?']


#creating features for test and train
X = df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']]
y = df[['Class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)

#creating our knn classifier model with n = 3
knn_value_3 = KNeighborsClassifier(n_neighbors = 3)
knn_value_3.fit(X_train, np.ravel(y_train,order='C'))

#Estimating the accuracy of knn classifier with 3 knn
print(knn_value_3.score(X_test, y_test))

#creating our knn classifier model with n = 5
knn_value_5 = KNeighborsClassifier(n_neighbors = 5)
knn_value_5.fit(X_train, np.ravel(y_train,order='C'))


#Estimating the accuracy of knn classifier with 5 knn
print(knn_value_5.score(X_test, y_test))

#creating our knn classifier model with n = 10
knn_value_10 = KNeighborsClassifier(n_neighbors = 10)
knn_value_10.fit(X_train, np.ravel(y_train,order='C'))

#Estimating the accuracy of knn classifier with 10 knn
print(knn_value_10.score(X_test, y_test))

#Making a confusion matrix
df_confusion_matrix = df.copy()
df_confusion_matrix = df_confusion_matrix.reset_index()
for i, row in df_confusion_matrix.iterrows():
    df_confusion_matrix.at[i, 'Predicted Class (k = 3)'] = knn_value_3.predict([[row['F1'], row['F2'], row['F3'], row['F4'], row['F5'], row['F6'], row['F7'], row['F8'], row['F9']]])[0]
    df_confusion_matrix.at[i, 'Predicted Class (k = 5)'] = knn_value_5.predict([[row['F1'], row['F2'], row['F3'], row['F4'], row['F5'], row['F6'], row['F7'], row['F8'], row['F9']]])[0]
    df_confusion_matrix.at[i, 'Predicted Class (k = 10)'] = knn_value_10.predict([[row['F1'], row['F2'], row['F3'], row['F4'], row['F5'], row['F6'], row['F7'], row['F8'], row['F9']]])[0]
    
df_confusion_matrix['Predicted Class (k = 3)'] =  df_confusion_matrix['Predicted Class (k = 3)'].astype(int)
df_confusion_matrix['Predicted Class (k = 5)'] =  df_confusion_matrix['Predicted Class (k = 5)'].astype(int)
df_confusion_matrix['Predicted Class (k = 10)'] =  df_confusion_matrix['Predicted Class (k = 10)'].astype(int)
#print(df_confusion_matrix)
df_confusion_matrix.to_csv("Confusion_Matrix.csv")






