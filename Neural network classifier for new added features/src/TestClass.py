# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix
from scipy.sparse import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph as kng
from scipy.sparse.linalg import expm
from scipy.spatial.distance import pdist
import scipy.spatial.distance
import math
import re
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense ,Dropout
from keras.utils import *
import numpy
import json
import codecs
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os,sys
from shutil import copyfile
from scipy.sparse import *
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import kneighbors_graph as kng
from scipy.sparse.linalg import expm
from scipy.linalg import solve_banded
import math
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


class KerasModel:
	
	def __init__(self,ModelDef,CompileFlags,TrainingOptions):
		self.ModelDef=ModelDef
		self.CompileFlags=CompileFlags
		self.Model=self.CreateModel()
		self.TrainingOptions=TrainingOptions
		
	def CreateModel(self):
		Model=Sequential()
		pattern=re.compile("mid.")
		Model.add(Dense(self.ModelDef["start"][0], input_dim=self.ModelDef["start"][3],kernel_initializer=self.ModelDef["start"][2], activation=self.ModelDef["start"][1]))			
		
		for key in self.ModelDef.keys():
			if pattern.match(key):
				Model.add(Dense(self.ModelDef[key][0], kernel_initializer=self.ModelDef[key][2], activation=self.ModelDef[key][1]))
		
		Model.add(Dense(self.ModelDef["end"][3], kernel_initializer=self.ModelDef["end"][2], activation=self.ModelDef["end"][1]))
		Model.compile(loss=self.CompileFlags["lossMethod"], optimizer=self.CompileFlags["optimizeMethod"], metrics=[self.CompileFlags["metrics"]])
		return Model
		
	def one_hot(self,input_array):
		unique,index=numpy.unique(input_array,return_inverse=True)
		return np_utils.to_categorical(index,len(unique))
		
	def FitModel(self,X_train,Y_train):
		out=self.one_hot(Y_train)
		self.Model.fit(X_train, out, epochs=self.TrainingOptions["epochs"], batch_size=self.TrainingOptions["batch_size"], verbose=0)
		
	def EvaluateModel(self,X_train,Y_train):
		out=self.one_hot(Y_train)
		scores = self.Model.evaluate(X_train, out, verbose=0)
		return scores
			
	def ModelPrediction(self,X_test):
		y_pred= self.Model.predict(X_test)
		return y_pred
	
	
class Training:
	
	def __init__(self,TrainingData,LaplacianScoreEstimator,seed=7):
		self.seed = seed
		self.dataset =numpy.loadtxt(TrainingData["DatasetName"], delimiter=",")
		self.n_splits=TrainingData["folds"]
		self.TrainTestRatio=TrainingData["TrainTestRatio"]
		self.dirname=TrainingData["ResultDirectory"]
		self.LaplacianScoreEstimator = LaplacianScoreEstimator
		#self.estimators={'SupervisedLearning':0,'GlobalUnSpuervisedLearning':1}
		self.estimators={'SupervisedLearning':self.KFoldValidation,'GlobalUnSupervisedLearning':self.LaplacianScoreEstimator.LaplacianScore,"UnsupervisedEvaluation":self.UnsupervisedEvaluation}
		
	def KFoldValidation(self,feature_number,Model,Semaphore):
	
		X_train,X_test,Y_train,Y_test = self.Projection(feature_number,self.TrainTestRatio);
		
		numpy.random.seed(self.seed)
		cvscores = []
		kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
		i=1
		os.chdir("../")
		pathname = os.getcwd()
		pathname = pathname+"/"+self.dirname+"/"
		os.chdir(pathname)
		dirname = "Result_"+str(feature_number)
		os.mkdir(dirname)
		os.chdir(pathname+dirname)
		filename = "Accuracy_Results_"+str(feature_number)
		text_file =open(filename,"w")
		fNo=str(feature_number)

		for train, test in kfold.split(X_train, Y_train):
			# Fit the Model
			Model.FitModel(X_train[train],Y_train[train])
			# evaluate the Model
			scores=Model.EvaluateModel(X_train[test],Y_train[test])
			text_file.write("Validation step " +str(i)+" of K fold Validation of "+fNo+" "+"%s: %.2f%%" % (Model.Model.metrics_names[1], scores[1]*100) )
			text_file.write("\n")
			cvscores.append(scores[1] * 100)
			i+=1			

		text_file.write("Overall Accuracy after k fold validation of "+fNo+" "+" %.2f%% (+/- %.2f%%)"  % (numpy.mean(cvscores), numpy.std(cvscores)) )
		text_file.write("\n")
		scores=Model.EvaluateModel(X_test,Y_test)
		text_file.write("Evaluated weight of feature " +fNo+" for recognition ""%s: %.2f%%" % (Model.Model.metrics_names[1], scores[1]*100))
		y_pred=Model.ModelPrediction(X_test)
		lx=self.dataset[:,feature_number-1:feature_number]
		LapScore = self.LaplacianScoreEstimator.LaplacianScore(X=lx)
		
		text_file.write("\n")
		#print LapScore[feature_number-1]
		
		text_file.write("%s"%(str(LapScore[feature_number-1])))
		c,ConfusionMatrix = self.CreateConfusionMatrix(y_pred,Model.one_hot(Y_test))
		cmatFile="ConfusionMatrixFeture"+fNo
		json.dump(ConfusionMatrix, codecs.open((cmatFile+".json"), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
		numpy.save(cmatFile,c)
		text_file.write("\n")
		text_file.close()
		self.LaplacianScoreEstimator.LapScores[fNo]=LapScore[feature_number-1]
		Semaphore.release()
		
		
		
	def CreateConfusionMatrix(self,y_pred,Y_test):
		Y_pred = numpy.argmax(y_pred,axis=1)
		c=confusion_matrix(numpy.argmax(Y_test,axis=1), Y_pred)
		ConfusionMatrix=c.tolist()
		return c,ConfusionMatrix
		
					
	def TargetDataColoumn(self):
		samples,target = self.dataset.shape
		return target
		
	
	def Projection(self,feature_Number , testSetSize):
		X= self.dataset[:,feature_Number-1:feature_Number]
		Y=self.dataset[:,self.TargetDataColoumn()-1]
		return train_test_split(X,Y,test_size=testSetSize,random_state=0)
	
	
	def UnsupervisedEvaluation(self,feature_number,X_selected,y,Semaphore,n_clusters=8):
		
		score = self.LaplacianScoreEstimator.LaplacianScore(X=X_selected)
		self.LaplacianScoreEstimator.LapScores[str(feature_number)]=score[feature_number-1]
		k_means = KMeans(n_clusters=n_clusters, init='k-means++', precompute_distances=True, verbose=0,random_state=None)
		k_means.fit(X_selected)
		y_predict = k_means.labels_
		y=y.flatten()
		# calculate NMI
		nmi = normalized_mutual_info_score(y, y_predict)
		Semaphore.release()
		
		
	
class LaplacianFeatureScore:
	
	def __init__(self,DataSetName,manager,**kwargs):
		
		self.DataSet = DataSetName
		X=numpy.loadtxt(self.DataSet,delimiter=',')
		s,n=X.shape
		self.X=X[:,0:n-1]
		
		if 'W' not in kwargs.keys():
			if 't_param' not in kwargs.keys():
				self.t_param=2
			else:
				self.t_param = kwargs['t_param']
		
			if 'neighbour_size' not in kwargs.keys():
				self.neighbour_size=16
			else:
				self.neighbour_size=kwargs['neighbour_size']
		# construct the affinity matrix W
			self.W = self.construct_W()
			self.n_samples, self.n_features = numpy.shape(self.X)
		
		else:
			self.W = kwargs['W']
    
		self.LapScores=manager.dict()
	
	def construct_W(self):
	
		k=self.neighbour_size
		t = self.t_param
		S=kng(self.X, k+1, mode='distance',metric='euclidean') #sqecludian distance works only with mode=connectivity  results were absurd
		S = (-1*(S*S))/(2*t*t)
		S=S.tocsc()
		S=expm(S)
		S=S.tocsr()
		
		
		#[1]  M. Belkin and P. Niyogi, “Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering,” Advances in Neural Information Processing Systems,
		#Vol. 14, 2001. Following the paper to make the weights matrix symmetrix we use this method
		 
		bigger = numpy.transpose(S) > S
		S = S - S.multiply(bigger) + numpy.transpose(S).multiply(bigger)
		return S

			
	def lap_score(self):
	
		# build the diagonal D matrix from affinity matrix W
		D = numpy.array(self.W.sum(axis=1))
		
		
		L = self.W
		
		
		tmp = numpy.dot(numpy.transpose(D),self.X)
		D = diags(numpy.transpose(D), [0])
		Xt = numpy.transpose(self.X)
		t1 = numpy.transpose(numpy.dot(Xt, D.todense()))
		t2 = numpy.transpose(numpy.dot(Xt, L.todense()))
		# compute the numerator of Lr
		tmp=numpy.multiply(tmp, tmp)/D.sum()
		D_prime = numpy.sum(numpy.multiply(t1, self.X), 0) -tmp 
		# compute the denominator of Lr
		L_prime = numpy.sum(numpy.multiply(t2, self.X), 0) -tmp 
		# avoid the denominator of Lr to be 0
		D_prime[D_prime < 1e-12] = 10000
		# compute laplacian score for all features
		score = 1 - numpy.array(numpy.multiply(L_prime, 1/D_prime))[0, :]
		return numpy.transpose(score)

	"""
		Rank features in ascending order according to their laplacian scores, the smaller the laplacian score is, the more
		important the feature is
	"""
	def feature_ranking(self,score):
		idx = numpy.argsort(score, 0)
		return idx+1

	def LaplacianScore(self,**kwargs):
		flag=0
		if 'X' not in kwargs.keys():
			X=self.X
			flag=1
		else:
			X=kwargs['X']
			
			
		#construct the diagonal matrix
		D=numpy.array(self.W.sum(axis=1))
		D = diags(numpy.transpose(D), [0])
		#construct graph Laplacian L
		L=D-self.W.toarray()
		
		#construct 1= [1,···,1]' 
		I=numpy.ones((self.n_samples,self.n_features))
		
		#construct fr' => fr= [fr1,...,frn]'
		Xt = numpy.transpose(X)
		
		#construct fr^=fr-(frt D I/It D I)I
		t=numpy.matmul(numpy.matmul(Xt,D.toarray()),I)/numpy.matmul(numpy.matmul(numpy.transpose(I),D.toarray()),I)
		t=t[:,0]
		t=numpy.tile(t,(self.n_samples,1))
		fr=self.X-t
		
		#Compute Laplacian Score
		fr_t=numpy.transpose(fr)
		Lr=numpy.matmul(numpy.matmul(fr_t,L),fr)/numpy.matmul(numpy.dot(fr_t,D.toarray()),fr)
		fScore = numpy.diag(Lr)
		k=1
		
		if flag==1:
			
			for i in fScore:
				self.LapScores[k]=i
				k=k+1
		else:
			return fScore
		
