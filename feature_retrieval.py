#!/usr/bin/python

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC, LinearSVC
import sys
import numpy
import os

output_dir = 'ModelNet10_large'
featureFile = output_dir + '/pool5.npy'
labelFile = output_dir + '/factors.txt'
numClasses = 10

if len(sys.argv) > 1:
	featureFile = sys.argv[1]

f = open(labelFile,'r')
labels = []
for l in f:
	labels.append([int(i) for i in l.split()])
f.close()
labels = numpy.array(labels)

data = numpy.load(featureFile)
totalPerClass = int(data.shape[0] / numClasses)
trainPerClass = int(totalPerClass * 0.8)
testPerClass = totalPerClass - trainPerClass
trainID=[]
testID=[]
for i in range(numClasses):
	trainID.extend(range(i*totalPerClass,i*totalPerClass+trainPerClass))
	testID.extend(range((i+1)*totalPerClass-testPerClass, (i+1)*totalPerClass))
	
features = {'train':data[trainID], 'test':data[testID]}
labels = {'train':labels[trainID], 'test':labels[testID]}

nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(features['train'])
dist, index = nbrs.kneighbors(features['test'])

examples = [0,16,32]
s = 'convert '
for i in range(len(examples)):
	j = testID[examples[i]]
	k1 = trainID[index[examples[i],0]]
	k2 = trainID[index[examples[i],1]]
	k3 = trainID[index[examples[i],2]]
	k4 = trainID[index[examples[i],3]]
	k5 = trainID[index[examples[i],4]]
	s += '\( %d.ppm -size 20x800 xc:none %d.ppm %d.ppm %d.ppm %d.ppm %d.ppm +append \) '  % (j,k1,k2,k3,k4,k5)
	s += '\( -size 4820x20 xc:none +append \) '
s += '-append ../report/examples.png'
print s

if output_dir == 'ModelNet10':
	factor=['class','orientation','fgColor','bgColor']
else:
	factor=['class','orientation','texture','bgColor']
for i in range(4):
	k_options = [1,3,5]
	acc={}
	for k in k_options:
		N = index.shape[0]
		idl = numpy.resize(index[:,:k].transpose(),N * k)
		truth = numpy.tile(labels['test'][:,i],k)
		match = truth == labels['train'][idl,i]
		if k > 1:
			match = numpy.any(numpy.resize(match,(k,N)).transpose(),axis=1)
		acc[k] = 1.0 * numpy.sum(match) / N
	print '%s & %.4f & %.4f & %.4f\\\\' % (factor[i],acc[k_options[0]],acc[k_options[1]],acc[k_options[2]])
