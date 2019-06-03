import multiprocessing as mp
from datetime import datetime
import time
import os
import numpy
import json
from TestClass import *




def CreateResultDirectory(Resdir):
	# Create Dir name by concatnating date and time
	os.chdir("../")
	
	dir_suffix=str(datetime.now().date())+"_"+str(datetime.now().time())
	dirname=Resdir+dir_suffix
	#Create dir
	os.mkdir(dirname)
	os.chdir("./src")
	return dirname

def GetDataSetFeatures(DataSetName):
	#Get the dataset name passed as a Command Line argument and extract no. of features from the dataset
	t=numpy.loadtxt(DataSetName, delimiter=",")
	samples ,number_of_features=t.shape;#(150,4)
	return number_of_features


def getCmdParameters(param):
	ResultDirectory = CreateResultDirectory(param.ResultDirectory)
	DataSetName,CompileFlags,ModelDef,TrainingOptions,LaplacianConfig,newDim= GetModelParam(param.ConfigFile)
	NoFeatures=GetDataSetFeatures(DataSetName)
	TrainingData={}		# Create an empty dictionary
	TrainingData["DatasetName"]=DataSetName  # Create a variable/key DatasetName and assign DatasetName value to it.
	TrainingData["folds"]=param.folds	# Create folds in dictionary and get folds from param and assign it to the folds in trainingdata
	TrainingData["TrainTestRatio"]=param.TrainTestRatio # create traintestratio and get it from param and assign it.
	TrainingData["ResultDirectory"]=ResultDirectory #ResultDictionary is already called from param in 1st line.
	return TrainingData,NoFeatures,CompileFlags,ModelDef,TrainingOptions,LaplacianConfig,newDim


def GetModelParam(ConfigFile):  # Initializing all the objects from the config file
	ModelDef={}
	CompileFlags={}
	TrainingOptions={}
	LaplacianConfig={}
	#print ConfigFile
	
	with open(ConfigFile) as json_file:
		data = json.load(json_file)
		
		for mD in data['ModelDescription']:
			ModelDef[mD["layer"]]=mD["no_of_neurons"], mD["activationFunction"], mD["initializationFunction"],mD["inputDimension"]
		
		for cF in data["CompileOption"]:
			CompileFlags["lossMethod"]=cF["lossMethod"]
			CompileFlags["optimizeMethod"]=cF["optimizeMethod"]
			CompileFlags["metrics"]=cF["metrics"]
		
		for dS in data["Dataset"]:  # When we want to access any object from the file we access it through loop,
			DataSetName=dS["name"]	# as we dont know how many objects are there
		
		
		for tOpt in data["TrainingOptions"]:
			TrainingOptions["epochs"]=tOpt["epochs"]
			TrainingOptions["batch_size"]=tOpt["batch_size"]

	
		for Ls in data["LaplacianScore"]:
			LaplacianConfig["neighbour_size"]=Ls["neighbour_size"]
			LaplacianConfig["t_param"]=Ls["t_param"]
			
		
		
			newDim=data["Dimensions"];
				
	return DataSetName,CompileFlags,ModelDef,TrainingOptions,LaplacianConfig, newDim

	
def SetupPlatform(param,Semaphore):
	TrainingData,NoFeatures,CompileFlags,ModelDef,TrainingOptions,LaplacianConfig,newDim=getCmdParameters(param)
	
	
	manager = mp.Manager()
	
	processList=[]
	n=LaplacianConfig["neighbour_size"]
	t_param=LaplacianConfig["t_param"]
	ls = LaplacianFeatureScore(TrainingData["DatasetName"],manager,neighbour_size=n, t_param=t_param)
	t=Training(TrainingData,ls)
	
	
	if param.method=="SupervisedLearning":
		for i in range (1,NoFeatures):
			Model=KerasModel(ModelDef,CompileFlags,TrainingOptions)
			p=mp.Process(target=t.estimators[param.method], args=(i,Model,Semaphore))
			processList.append(p)
		
	if param.method=="GlobalUnSupervisedLearning":
		p = mp.Process(target=t.estimators[param.method],args=())
		processList.append(p)
		
	if param.method=="UnsupervisedEvaluation":
		X=numpy.loadtxt(TrainingData["DatasetName"],delimiter=",")
		row,col=X.shape
		y=X[:,col-1:col]
		X=X[:,0:col]
		row,col=X.shape
		X_selected=X[:,col-newDim-1:col-1]
		
		for i in range(col-newDim-1,col-1):
			p=mp.Process(target=t.estimators["UnsupervisedEvaluation"], args=(i,X[:,i:i+1],y,Semaphore))
			processList.append(p)
	
	
	return processList,ls
			

