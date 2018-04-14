import numpy as np

Tr = np.genfromtxt('usps-4-9-train.csv' , delimiter=',')

training_rate = 0.001

#print Tr[0][256]

trX= np.array(Tr[:,:256])
trY= np.array(Tr[:,256:257])

#print len(trX)

w = []
for i in range(len(trX[0])):
	
	
	w.append(0)


for i in range(100):
	g = []
	for i in range(len(trX[0])):
	
		g.append(0)
	
	
	for i in range(len(trX)):

		Yi = 1/(1 + np.exp(-1* np.dot(w,i)))
		#print 1/(1 + np.exp(-1* np.dot(w,i)))
		g = np.add(g, np.multiply((Yi - trY[i]), trX[i]))
		#print np.multiply((Yi - trY[i]), trX[i])
		
	w = w - np.multiply(training_rate, g)
print w
	

