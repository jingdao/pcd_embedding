import numpy
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import NearestNeighbors
import time
import os
import sys
os.environ['GLOG_minloglevel']="1"
import caffe
caffe.set_mode_gpu()

output_dir = 'E:/jd/Documents/pcd_embedding/ModelNet10_test'
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
nets = {
	'class': caffe.Net('architecture/pool5_class.prototxt','architecture/class.caffemodel',caffe.TEST),
	'orientation': caffe.Net('architecture/pool5_orientation.prototxt','architecture/orientation.caffemodel',caffe.TEST),
	'bgColor': caffe.Net('architecture/pool5_color.prototxt','architecture/bgColor.caffemodel',caffe.TEST)
}
if output_dir.endswith('ModelNet10') or output_dir.endswith('ModelNet10_test'):
	factor=['class','orientation','fgColor','bgColor']
	nets['fgColor'] = caffe.Net('architecture/pool5_color.prototxt','architecture/fgColor.caffemodel',caffe.TEST)
else:
	factor=['class','orientation','texture','bgColor']
	nets['texture'] = caffe.Net('architecture/pool5_class.prototxt','architecture/texture.caffemodel',caffe.TEST)
	
def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()
	#return x
nbrs = [None] * len(factor)
index = [None] * len(factor)
for i in range(len(factor)):
	if factor[i] not in nets:
		continue
	f1 = []
	f2 = []
	nets[factor[i]].blobs['data'].reshape(1,9216)
	if factor[i] in ['class','texture','orientation']:
		nets[factor[i]].blobs['label'].reshape(1)
	#elif factor[i] == 'orientation':
		#nets[factor[i]].blobs['label'].reshape(1,2)
	else:
		nets[factor[i]].blobs['label'].reshape(1,3)
	layerID = 'fc3' if factor[i] == 'orientation' else 'fc1'
	for j in range(len(features['train'])):
		nets[factor[i]].blobs['data'].data[0,:] = features['train'][j]
		nets[factor[i]].forward()
		if factor[i] in ['class','texture','orientation']:
			f1.append(softmax(numpy.array(nets[factor[i]].blobs[layerID].data[0,:])))
		else:
			f1.append(numpy.array(nets[factor[i]].blobs[layerID].data[0,:]))
	for j in range(len(features['test'])):
		nets[factor[i]].blobs['data'].data[0,:] = features['test'][j]
		nets[factor[i]].forward()
		if factor[i] in ['class','texture','orientation']:
			f2.append(softmax(numpy.array(nets[factor[i]].blobs[layerID].data[0,:])))
		else:
			f2.append(numpy.array(nets[factor[i]].blobs[layerID].data[0,:]))
	nbrs[i] = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(f1)
	dist, index[i] = nbrs[i].kneighbors(f2)

for i in range(len(factor)):
	if index[i] is None:
		continue
	k_options = [1,3,5]
	acc={}
	for k in k_options:
		N = index[i].shape[0]
		idl = numpy.resize(index[i][:,:k].transpose(),N * k)
		truth = numpy.tile(labels['test'][:,i],k)
		match = truth == labels['train'][idl,i]
		if k > 1:
			match = numpy.any(numpy.resize(match,(k,N)).transpose(),axis=1)
		acc[k] = 1.0 * numpy.sum(match) / N
	print '%s & %.4f & %.4f & %.4f\\\\' % (factor[i],acc[k_options[0]],acc[k_options[1]],acc[k_options[2]])
