__author__ = "Luiz Gustavo Silveira Rossa"
__version__ = "2.0"

#matrizes de treinamento geradas a partir de imagens capturadas por uma webcam.

import numpy as np
from sklearn import datasets, svm, metrics, model_selection
import pickle
from sklearn import svm
from sklearn import datasets
from matplotlib import style
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import os, glob

def acuracia(clf,X,y):
   resultados = cross_val_predict(clf, X, y, cv=5)
   return accuracy_score(y,resultados)


#Classificador do SVM
classifier = svm.SVC(gamma=0.001, kernel = 'linear')

vetor_semVinco = []
vetor_comVinco = []
vetor_mesa = []
vetor_semVinco_alto_brilho = []

#diretorio do projeto
old = os.getcwd()

def get_matrix(local_file, vector):
	os.chdir(local_file)
	for file in glob.glob('*.txt'):
	    for line in open(file, 'r'):
	    	aux = [line]
	    	matrix_file = np.loadtxt(aux, delimiter = ",")
	    	vector.append(matrix_file) 
	#retorna para o diretorio do projeto  	
	os.chdir(old)

#chama a funcao responsavel por carregar todas as matrizes de dados que estao separadas em diretorios especificos
get_matrix("comVinco/", vetor_semVinco)
get_matrix("semVinco/", vetor_comVinco)
get_matrix("mesa/", vetor_mesa)


#concatena as matrizes em uma unica lista para utilizar no treinamento
samples = vetor_semVinco + vetor_comVinco + vetor_mesa

#VETOR DE TREINAMENTO
#1 - SEM VINCO
#2 - COM VINCO
#3 - MESA DA LINHA
#4 - SEM VINCO BOBINA COLORIDA ALTO BRILHO

targetTraining = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
				  3, 3, 3, 3, 3, 3, 3, 3, 3, 3]


print("----------------------")
print("VETORES DE TREINAMENTO")
print("vetor 1 - Conjunto de dados sem vinco")
print("vetor 2 - Conjunto de dados com vinco")
print("vetor 3 - Conjunto de dados da mesa da linha")
print("vetor 4 - Conjunto de dados sem vinco bobina colorida com alto brilho")
print("----------------------")



#REALIZA O TREINAMENTO DO SCRIPT
classifier.fit(samples, targetTraining)
#print("Precision: ")
#print(acuracia(classifier,samples,targetTraining))

#calcula a precisao do algoritmo para os 20% da base de dados nao utilizados no treinamento
print("Predict")
predict = classifier.predict(samples)
print(predict)
print("--------------")
#print(classifier.predict(samples))


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(targetTraining, predict)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(targetTraining, predict))

#salva um arquivo com a base de treinamento no diretorio do projeto
saveFile = 'svm_treinado_matriz.sav'
pickle.dump(classifier, open(saveFile, 'wb'))