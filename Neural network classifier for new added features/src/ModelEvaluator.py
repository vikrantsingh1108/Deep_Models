#python ModelEvaluator.py -f ../Config/Iris.json -d ResultIrisDataset -t 0.2 -k 3 -m 2
#python ModelEvaluator.py -f ../Config/Diabities.json -d ResultDiabitiesDataset -t 0.2 -k 3 -m 2

#!/usr/bin/python
# -*- coding: utf-8 -*-

from AuxFunctions import *
import multiprocessing as mp

import argparse as aP

def ArgParser():
	parser = aP.ArgumentParser(description='Help for Command line arguments')
	parser.add_argument('-f', action="store", dest="ConfigFile" , help="Configuration File Path with Name eg /path_to_file/filename.json")
	parser.add_argument('-d', action="store", dest="ResultDirectory",help="Directory where the Result of Neural network Model Validation are stored")
	parser.add_argument('-t', action="store", dest="TrainTestRatio" ,type=float , help="Parameter specifying The Ratio to split the Dataset in test and Training")
	parser.add_argument('-k', action="store", dest="folds" , type=int , help = "Specify the number of folds for K-Cross Fold Validation")
	parser.add_argument('-c', action="store", dest="maxConn" , type=int , help="Specify the maximum number of process in parallel which this program is allowed to spawn")
	parser.add_argument('-m', action="store", dest="method" ,  help="learning Method")
	
	argParams = parser.parse_args()	
	return argParams
		
if __name__=='__main__':
	param =  ArgParser()
	Semaphore = mp.Semaphore(param.maxConn)
	
	"""
	TrainingData,NoFeatures,CompileFlags,ModelDef,TrainingOptions,LaplacianConfig=getCmdParameters(param)
	
	
	manager = mp.Manager()
	
	processList=[]
	n=LaplacianConfig["neighbour_size"]
	t_param=LaplacianConfig["t_param"]
	ls = LaplacianFeatureScore(TrainingData["DatasetName"],manager,neighbour_size=n, t_param=t_param)
	t=Training(TrainingData,ls)
	
	#Ls=ls.LaplacianScore()
	#print ls.feature_ranking(Ls)
	
	
	for i in range (1,NoFeatures):
		Model=KerasModel(ModelDef,CompileFlags,TrainingOptions)
		p=mp.Process(target=t.estimate, args=(i,Model,Semaphore,t.estimators['SupervisedLearning']))
		processList.append(p)
	"""	
	processList,estimate=SetupPlatform(param,Semaphore)
	
	for p in processList:
		Semaphore.acquire()
		p.start()
	
	for p in processList:
		p.join()

		
	print estimate.LapScores
	print "end"
	
	



	

	
