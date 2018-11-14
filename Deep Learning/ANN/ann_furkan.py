#Ann
# Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_X_1 = LabelEncoder()
encoder_X_2 = LabelEncoder()
X[:,1] = encoder_X_1.fit_transform(X[:,1])
X[:,2] = encoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
  
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Make the ANN

# Importing Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
      
# Initializing the ANN
classifier = Sequential()

# Adding input layer and first hidden kayer
classifier.add(Dense(6, activation = 'relu', kernel_initializer = 'uniform', input_dim=11))

# Adding second hidden layer
classifier.add(Dense(6, activation = 'relu', kernel_initializer = 'uniform'))  

# Adding output layer  
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'uniform')) 

# Compling the ANN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)