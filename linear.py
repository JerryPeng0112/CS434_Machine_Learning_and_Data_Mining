import numpy as np
from numpy.linalg import inv

def linearRegressionTraining():
	file = open("housing_train.txt", "r")
	fileStr = file.read()
	file.close()
	fileStrs = fileStr.split('\n')

	#clear empty lines
	fileStrs = [elem for elem in fileStrs if elem != ""]
			
	# clear empty strings
	X = []
	Y = []
	for i in fileStrs:
		tempDataStr = i.split(" ")
		tempDataStr = [float(elem) for elem in tempDataStr if elem != ""]
		X.append([1] + tempDataStr[0:-1])
		Y.append(tempDataStr[-1])

	X = np.matrix(X)
	Y = np.transpose(np.matrix(Y))
	Xt = np.transpose(X)
	XtX = np.matmul(Xt,X)
	XtXinv = inv(XtX)
	XtY = np.matmul(Xt, Y)
	w = np.matmul(XtXinv, XtY)
	print "Weight vectors:"
	print w
	return w

def calcTrainingSSE(w):
	files = ["housing_train.txt", "housing_test.txt"]
	w = np.transpose(w)
	for index in range(2):
		file = open(files[index], "r")
		fileStr = file.read()
		fileStrs = fileStr.split('\n')
		
		#clear empty lines
		fileStrs = [elem for elem in fileStrs if elem != ""]
		
		# clear empty strings
		X = []
		Y = []
		for i in fileStrs:
			tempDataStr = i.split(" ")
			tempDataStr = [float(elem) for elem in tempDataStr if elem != ""]
			X.append([1] + tempDataStr[0:-1])
			Y.append(tempDataStr[-1])
		
		#calculate normalized SSE
		predictedY = []
		
		for i in X:
			temp = np.matrix(i)
			temp = np.transpose(temp)
			predictedY.append(np.matmul(w, temp))
		
		sum = 0;
		for i in range(0, len(Y)):
			sum= sum + ((Y[i] - predictedY[i])*(Y[i] - predictedY[i]))
			
		n = len(Y)
		print sum/n
		
def linearRegressionTrainingNoDummy():
	file = open("housing_train.txt", "r")
	fileStr = file.read()
	file.close()
	fileStrs = fileStr.split('\n')

	#clear empty lines
	fileStrs = [elem for elem in fileStrs if elem != ""]
			
	# clear empty strings
	X = []
	Y = []
	for i in fileStrs:
		tempDataStr = i.split(" ")
		tempDataStr = [float(elem) for elem in tempDataStr if elem != ""]
		X.append(tempDataStr[0:-1])
		Y.append(tempDataStr[-1])

	X = np.matrix(X)
	Y = np.transpose(np.matrix(Y))
	Xt = np.transpose(X)
	XtX = np.matmul(Xt,X)
	XtXinv = inv(XtX)
	XtY = np.matmul(Xt, Y)
	w = np.matmul(XtXinv, XtY)
	print "Weight vectors:"
	print w
	return w

def calcNoDummyTrainingSSE(w):
	files = ["housing_train.txt", "housing_test.txt"]
	w = np.transpose(w)
	for index in range(2):
		file = open(files[index], "r")
		fileStr = file.read()
		fileStrs = fileStr.split('\n')
		
		#clear empty lines
		fileStrs = [elem for elem in fileStrs if elem != ""]
		
		# clear empty strings
		X = []
		Y = []
		for i in fileStrs:
			tempDataStr = i.split(" ")
			tempDataStr = [float(elem) for elem in tempDataStr if elem != ""]
			X.append(tempDataStr[0:-1])
			Y.append(tempDataStr[-1])
		
		#calculate normalized SSE
		predictedY = []
		
		for i in X:
			temp = np.matrix(i)
			temp = np.transpose(temp)
			predictedY.append(np.matmul(w, temp))
		
		sum = 0;
		for i in range(0, len(Y)):
			sum= sum + ((Y[i] - predictedY[i])*(Y[i] - predictedY[i]))
			
		n = len(Y)
		print sum/n
	
def main():
	w = linearRegressionTraining()
	calcTrainingSSE(w)
	w = linearRegressionTrainingNoDummy()
	calcNoDummyTrainingSSE(w)
	
	
if __name__ == "__main__":
	main()
	