#import all libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
#import dataset:
dataset=pd.read_csv('Churn_Modelling.csv')
# Spliting the dataset into Depependent and indepepdnet:
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encodeing some values:
encoder1=LabelEncoder()
X[:,1]=encoder1.fit_transform(X[:,1])
encoder2=LabelEncoder()
X[:,2]=encoder2.fit_transform(X[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Spliting the dataset into train and test:
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Feature Scaling:
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

# Initialization of ANN:
classifier=Sequential()

# Adding the Input_layer and First_hidden_layer:
classifier.add(Dense(units=6,
                     kernel_initializer='uniform',
                     activation='relu',
                     input_dim=11))
#classifier.add(Dropout(p=0.1))
# Adding the Hidden_layer:
classifier.add(Dense(uints=6,
                     kernel_initializer='uniform',
                     activation='relu'))
#classifier.add(Dropout(p=0.1))

# Adding the output_layer:
classifier.add(Dense(units=1,
                     kernel_initializer='uniform',
                     activation='sigmoid'))

#Compile the ANN:
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Fitting the ANN to the Training:
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Making the prediction:
y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5)

# Making the cofusion matrix:
cm=confusion_matrix(y_test,y_pred)

# Evaluating the ANN:
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracy=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
mean=accuracy.mean()
variance=accuracy.std()

# droupout regulization to remove  the overfitting: 
# Tunning the parametres:
def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uinform',activation='relu',input_dim=11))
    classifier.add(Dense(units=6,kernel_initializer='uinform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uinform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,30],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']}

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_
