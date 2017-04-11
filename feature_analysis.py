#!/usr/bin/python

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
import sys
import numpy
import os

output_dir = 'ModelNet10'
featureFile = output_dir + '/pool5.npy'
labelFile = output_dir + '/factors.txt'
numClasses = 10

if len(sys.argv) > 1:
	featureFile = sys.argv[1]

f = open(labelFile,'r')
labelID = []
for l in f:
	labelID.append([int(i) for i in l.split()])
f.close()
labelID = numpy.array(labelID)

data = numpy.load(featureFile)
totalPerClass = int(data.shape[0] / numClasses)
trainPerClass = int(totalPerClass * 0.8)
testPerClass = totalPerClass - trainPerClass
trainID=[]
testID=[]
for i in range(numClasses):
	trainID.extend(range(i*totalPerClass,i*totalPerClass+trainPerClass))
	testID.extend(range((i+1)*totalPerClass-testPerClass, (i+1)*totalPerClass))
	
#import data into matrix format
features = {'train':data[trainID], 'test':data[testID]}
labels = {
	'class': {'train':labelID[trainID,0], 'test':labelID[testID,0]},
	'orientation': {'train':labelID[trainID,1], 'test':labelID[testID,1]},
	'bgColor': {'train':labelID[trainID,3], 'test':labelID[testID,3]},
}
if output_dir=='ModelNet10':
	labels['fgColor'] = {'train':labelID[trainID,2], 'test':labelID[testID,2]}
else:
	labels['texture'] = {'train':labelID[trainID,2], 'test':labelID[testID,2]}
	
#print 'Using features from',featureFile,'...'
for factor in labels.keys():
	if factor=='orientation':
		knn1score=[]
		knn5score=[]
		svm_score=[]
		numTrain = int(features['train'].shape[0] / numClasses)
		numTest = int(features['test'].shape[0] / numClasses)
		for i in range(numClasses):
			t1 = features['train'][i*numTrain : (i+1)*numTrain]
			l1 = labels[factor]['train'][i*numTrain : (i+1)*numTrain]
			t2 = features['test'][i*numTest : (i+1)*numTest]
			l2 = labels[factor]['test'][i*numTest : (i+1)*numTest]
			neigh = KNeighborsClassifier(1,p=1,weights='distance')
			neigh.fit(t1,l1)
			knn1score.append(neigh.score(t2,l2))
			neigh = KNeighborsClassifier(1,p=5,weights='distance')
			neigh.fit(t1,l1)
			knn5score.append(neigh.score(t2,l2))
			svc = LinearSVC(random_state=0)
			svc.fit(t1,l1)
			svm_score.append(svc.score(t2,l2))
#		print 'KNN=1 %s Accuracy %.4f' % (factor,numpy.mean(knn1score))
#		print 'KNN=5 %s Accuracy %.4f' % (factor,numpy.mean(knn5score))
#		print 'SVM %s Accuracy %.4f' % (factor,numpy.mean(svm_score))
		print '%s & %.4f & %.4f & %.4f\\\\' % (factor,numpy.mean(knn1score),numpy.mean(knn5score),numpy.mean(svm_score))
		continue

	neigh = KNeighborsClassifier(1,p=1,weights='distance')
	neigh.fit(features['train'],labels[factor]['train'])
	knn1score = neigh.score(features['test'],labels[factor]['test'])
#	print 'KNN=1 %s: Accuracy %.4f' % (factor, neigh.score(features['test'],labels[factor]['test']))

	neigh = KNeighborsClassifier(5,p=1,weights='distance')
	neigh.fit(features['train'],labels[factor]['train'])
	knn5score = neigh.score(features['test'],labels[factor]['test'])
#	print 'KNN=5 %s: Accuracy %.4f' % (factor, neigh.score(features['test'],labels[factor]['test']))

	svc = LinearSVC(random_state=0)
	svc.fit(features['train'],labels[factor]['train'])
	svm_score = svc.score(features['test'],labels[factor]['test'])
#	print 'SVM %s: Accuracy %.4f' % (factor, svc.score(features['test'],labels[factor]['test']))

	print '%s & %.4f & %.4f & %.4f\\\\' % (factor,knn1score,knn5score,svm_score)
