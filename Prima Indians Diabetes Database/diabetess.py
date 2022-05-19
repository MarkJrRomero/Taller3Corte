import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

simplefilter(action='ignore', category=FutureWarning)

datos = pd.read_csv('diabetes.csv')

datos.Age.replace(np.nan, 33, inplace=True)

rangos = [20, 35, 50, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']

datos.Age = pd.cut(datos.Age, rangos, labels=nombres)
datos.drop(['DiabetesPedigreeFunction', 'BMI', 'Insulin', 'BloodPressure'], axis=1, inplace=True)


train_data = datos[:384]
test_data = datos[384:]


x = np.array(train_data.drop(['Outcome'], 1))
y = np.array(train_data.Outcome) 


train_x, test_x, train_y, test_y = train_test_split(x, y, size_test=0.2)

test_out_x = np.array(test_data.drop(['Outcome'], 1))
test_out_y = np.array(test_data.Outcome) 

#  Algoritmos
