
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


#%%
df = pd.read_csv('wisc_bc_ContinuousVar.csv')
df.drop('id',inplace= True,axis=1)
df.info()


#%%
#Using label encoder to label all categorical variables
le = LabelEncoder()
for column in df:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])
df.head()


#%%
X = df.drop(['diagnosis'], axis=1)
y = df[['diagnosis']]
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)
X_train.shape, X_test.shape


#%%
clf_svm = make_pipeline(StandardScaler(), SVC()).fit(X_train, np.ravel(y_train,order='C'))

print("Accuracy score ->", clf_svm.score(X_test, y_test))

#creating confusion matrix
y_pred = clf_svm.predict(X_test)
cm=confusion_matrix(y_test, y_pred)

print('Confusion matrix\n', cm)
print('True Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])

# %%
