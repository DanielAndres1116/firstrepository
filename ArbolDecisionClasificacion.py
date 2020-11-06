# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:48:30 2020

@author: Daniel Andres
"""

############ LIBRERÍAS A UTILIZAR ##################

#Se importan las librerías a utilizar
from sklearn import datasets


########### PREPARAR LA DATA ######################

#Importamos los datos de la misma libreria de scikit-learn
dataset = datasets.load_breast_cancer()
print(dataset)


########### ENTENDIMIENTO DE LA DATA ##############

#Verifico la información contenida en el dataset
print('Información en el dataset: ')
print(dataset.keys())
print()

#Verifico las caracteristicas del dataset
print('Características del dataset: ')
print(dataset.DESCR)

#Seleccionamos todas las columnas
X = dataset.data

#Defino los datos correspondientes a las etiquetas
y = dataset.target


################### IMPLEMENTACIÓN DE NAIVE BAYES ###############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Defino el algoritmo a utilizar
#Arboles de decision
from sklearn.tree import DecisionTreeClassifier
algoritmo = DecisionTreeClassifier(criterion = 'entropy') #default = 'gini'

#Entreno el modelo
algoritmo.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo.predict(X_test)

#Verifico la matriz de confusión
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión')
print(matriz)

#Calculo la precisión del modelo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precisión del modelo: ')
print(precision)


