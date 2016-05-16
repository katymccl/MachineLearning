import argparse
import numpy as np
import pickle
import resource
from scipy.sparse import coo_matrix


def args():
	parser = argparse.ArgumentParser(description = 'Convert text data into pickled data for 497 Final')
	parser.add_argument('-tX', '--trainX', 	default="", help='Training Data')
	parser.add_argument('-tT', '--trainT', 	default="", help='Training Targets')
	parser.add_argument('-dX', '--devX', 	default="", help='Development Data')
	parser.add_argument('-dT', '--devT', 	default="", help='Development Targets')
	parser.add_argument('-c', '--config',	default="", help='Config File')
	parser.add_argument('-d', '--directory', default="", help='Directory of data files. Will be prepended onto the given data/config files. If none is given, full paths must be given for data/config files.')

	args = parser.parse_args()
	return args.trainX, args.trainT, args.devX, args.devT, args.config, args.directory

def main():
	trainX, trainT, devX, devT, config, directory = args()
	resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
	if(config == ""):
		print("No Config File as given. Cannot create datasets without one.")
		exit()
	else:
		CONFIG = open(directory+config)
		lines = CONFIG.readlines()
		for line in lines:
			conf = line.split()
			if(conf[0] == "N_TRAIN"):
				NTRAIN = int(conf[1])
			if(conf[0] == "N_DEV"):
				NDEV = int(conf[1])
			if(conf[0] == "D"):
				D = int(conf[1])
			if(conf[0] == "C"):
				C = int(conf[1])

	if((trainX == "") and (trainT == "") and (devX == "") and (devT == "")):
		print("No data files given.")
		exit()

	if(not(trainX == "")):
		TRAIN_X = open(directory+trainX)
		traindatarow  = np.empty(NTRAIN)
		traindatacolumn = np.empty(NTRAIN)
		traindata = np.empty(NTRAIN)



		for i, ir in enumerate(TRAIN_X):
			temp = np.asarray(ir.split(), dtype=np.float)
			traindatarow = np.append(traindatarow, temp[0])
			traindatacolumn = np.append(traindatacolumn,temp[1])
			traindata = np.append(traindata,temp[2])

		tdata = coo_matrix((traindata, (traindatarow, traindatacolumn)), shape=(NTRAIN, D)).toarray()
		


		
		#pickle.dump(tdata, open("trainX.p", "wb"))

	if(not(trainT == "")):
		TRAIN_T = open(directory+trainT)
	
		traintarget = np.empty((NTRAIN,), dtype=np.int)

		for i, ir in enumerate(TRAIN_T):
			traintarget[i] = np.asarray(ir, dtype=np.int)
		#pickle.dump(traintarget, open("trainT.p", "wb"))


	if(not(devX == "")):
		DEV_X = open(directory+devX)
	
		devdatarow  = np.empty(NTRAIN)
		devdatacolumn = np.empty(NTRAIN)
		devdata = np.empty(NTRAIN)

		for i, ir in enumerate(DEV_X):
			temp2 = np.asarray(ir.split(), dtype=np.float)
			devdatarow = np.append(devdatarow, temp[0])
			devdatacolumn = np.append(devdatacolumn,temp[1])
			devdata = np.append(devdata,temp[2])
		#pickle.dump(devdata, open("devX.p", "wb"))


	if(not(devT == "")):
		DEV_T = open(directory+devT)
	
		devtarget = np.empty((NDEV,), dtype=np.int)

		for i, ir in enumerate(DEV_T):
			devtarget[i] = np.asarray(ir, dtype=np.int)
		#pickle.dump(devtarget, open("devT.p", "wb"))



	print(devtarget)
	print(traintarget)

main()
