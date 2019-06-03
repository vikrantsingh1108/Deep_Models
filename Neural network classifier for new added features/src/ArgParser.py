import argparse as aP

def ArgParser():
	parser = aP.ArgumentParser(description='Parse the Command line arguments')
	parser.add_argument('-f', action="store", dest="ConfigFile")
	parser.add_argument('-d', action="store", dest="ResultDirectory")
	parser.add_argument('-t', action="store", dest="TrainTestRatio" ,type=float)
	parser.add_argument('-k', action="store", dest="folds" , type=int)
	parser.add_argument('-m', action="store", dest="maxConn" , type=int)
	argParams = parser.parse_args()	
	return argParams


	
