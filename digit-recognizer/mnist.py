#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer
# ### Learn computer vision fundamentals with the famous MNIST data
# 
# 

# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


data_train = pd.read_csv('train.csv')
x_train = data_train.drop(['label'], axis=1)
y_train = data_train['label']

print(x_train.shape)
print(y_train.shape)

x_train_norm = x_train / 255.0 # normalize the data
print(x_train_norm.iloc[0:10, 220:240])
# y_train_0 = (y_train == 0) # -> binary classification, if sample is '0' or not


# In[3]:


x_test = pd.read_csv('test.csv')  
x_test = x_test / 255.0


# ## Linear classifiers (SVM, logistic regression, etc.) with SGD training.
# 
# 'SGDClassifier' w scikit-learn to klasa implementująca algorytm SGD (Stochastic Gradient Descent) do problemów klasyfikacji. Jest to klasyfikator liniowy, co oznacza, że szuka hiperpłaszczyzny dzielącej przestrzeń cech, aby oddzielić różne klasy.
# 
# SGD jest algorytmem optymalizacji stosowanym do minimalizacji funkcji kosztu, a w kontekście SGDClassifier funkcja kosztu jest związana z modelem liniowym. Algorytm ten aktualizuje wagi modelu w sposób stochastyczny, tj. po jednym przykładzie uczącym na raz, co jest korzystne dla dużych zbiorów danych.
# 

# In[6]:


sgd = SGDClassifier(random_state=42)

sgd.fit(x_train_norm, y_train)

y_train_pred = sgd.predict(x_train_norm)
accuracy_train_sgd = accuracy_score(y_train, y_train_pred)
print(accuracy_train_sgd)


# In[7]:


y_test_kaggle = sgd.predict(x_test)
y_test_kaggle


# To help with visualizing data I plot it.

# In[8]:


x = 20
for i in range(x):
    image = np.array(x_train.iloc[i]).reshape(28, 28)

    plt.subplot(4, int(x / 4), i + 1)  # 2 rows, 5 columns
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.title(f"Image {i + 1}")

first_20_predictions = sgd.predict(x_train_norm.iloc[0:20,])
print(first_20_predictions)
plt.show()


# The accuracy of SGDClassifier is quite low - only 92%. Let me try different methods

# ## SVC - C-Support Vector Classification.

# In[6]:


parameters = {'C':[1, 10, 100]}

svc_clf = SVC(random_state=42)

grid_svc_clf = GridSearchCV(svc_clf, parameters)

grid_svc_clf.fit(x_train_norm, y_train)

print(f"Best parameters: {grid_svc_clf.best_params_}")

best_svc_clf = grid_svc_clf.best_estimator_

y_train_predict_svc = best_svc_clf.predict(x_train_norm)

accuracy_train_svc_clf = accuracy_score(y_train, y_train_predict_svc)
print(accuracy_train_svc_clf)


# In[4]:


# polynomial_svm_clf = Pipeline([
#     ("poly_features", PolynomialFeatures(degree=3,include_bias=False)),
#     ("scaler", StandardScaler()),
#     ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=50))])
polynomial_svm_clf = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=50))])
polynomial_svm_clf.fit(x_train_norm, y_train)

y_train_predict_pol_svm = polynomial_svm_clf.predict(x_train_norm)

accuracy_train_svc_pol_clf = accuracy_score(y_train, y_train_predict_pol_svm)
print(accuracy_train_svc_pol_clf)


# Saving SVC and Polynomial SVC classifier results

# In[8]:


# SVC
results_svc = best_svc_clf.predict(x_test)
results_svc_df = pd.DataFrame({'Label': results_svc})
results_svc_df['ImageId'] = results_svc_df.index + 1
results_svc_df = results_svc_df[['ImageId', 'Label']]
results_svc_df.to_csv('results_svc_classifier.csv', index=False)

# Polynomial SVC
results_pol_svc = polynomial_svm_clf.predict(x_test)
results_pol_svc_df = pd.DataFrame({'Label': results_pol_svc})
results_pol_svc_df['ImageId'] = results_pol_svc_df.index + 1
results_pol_svc_df = results_pol_svc_df[['ImageId', 'Label']]
results_pol_svc_df.to_csv('results_pol_svc_classifier.csv', index=False)


# Trying with MLP-classifier:
# - splitting labeled data into train and validation set
# - creating one-hot-encoding
# - adding layers, compiling the model and fitting it with the data

# In[12]:


x_train_2, x_valid, y_train_2, y_valid = train_test_split(x_train_norm, y_train, test_size=0.2, random_state=42)

y_train_2 = to_categorical(y_train_2, num_classes=10) 
y_valid = to_categorical(y_valid, num_classes=10)

mlp_clf = keras.models.Sequential()

mlp_clf.add(tf.keras.Input(shape=(784,)))
mlp_clf.add(keras.layers.Dense(256, activation="relu"))
mlp_clf.add(keras.layers.Dense(128, activation="relu"))
mlp_clf.add(keras.layers.Dense(128, activation="relu"))
mlp_clf.add(keras.layers.Dense(128, activation="relu"))
mlp_clf.add(keras.layers.Dense(128, activation="relu"))
mlp_clf.add(keras.layers.Dense(32, activation="relu"))
mlp_clf.add(keras.layers.Dense(10, activation="softmax"))

mlp_clf.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

print(mlp_clf.summary())

history = mlp_clf.fit(x_train_2, y_train_2, epochs=30, validation_data=(x_valid, y_valid), batch_size=10)
            
history


# Saving mlp classfier results:

# In[13]:


results_mlp = mlp_clf.predict(x_test)

result_mlp_df = pd.DataFrame({'Label': results_mlp.argmax(axis=1)})

result_mlp_df['ImageId'] = result_mlp_df.index + 1

result_mlp_df = result_mlp_df[['ImageId', 'Label']]

result_mlp_df.to_csv('results_mlp_classifier.csv', index=False)


# Visualize the data:

# In[11]:


x = 60
for i in range(x):
    image = np.array(x_test.iloc[i]).reshape(28, 28)

    plt.subplot(4, int(x / 4), i + 1)  # 2 rows, 5 columns
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.title(f"Image {i + 1}")

plt.show() 

result_mlp_df.head(20)


# In[ ]:




