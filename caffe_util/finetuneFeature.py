import numpy
import matplotlib.pyplot as plt
import scipy
import time
import os
import sys
os.environ['GLOG_minloglevel']="1"
import caffe
caffe.set_mode_gpu()

#palette = [
#	[10,10,10], 
#	[255,255,255],
#	[255,10,10], 
#	[10,255,10],
#	[10,10,255], 
#	[255,255,10],
#	[255,10,255],
#	[10,255,255],
#	[255,128,10],
#	[128,10,255],
#]

palette = [
	[50,50,50],
	[147, 147, 147],
	[254,   9,   9],
	[  9, 254,   9],
	[  9,   9, 254],
	[180, 180,   7],
	[180,   7, 180],
	[  7, 180, 180],
	[227, 114,   8],
	[114,   8, 227],
]

#feature_type = 'class'
#feature_type = 'bgColor'
#feature_type = 'fgColor'
#feature_type = 'orientation'
feature_type = 'texture'
output_dir = 'E:/jd/Documents/pcd_embedding/ModelNet10_large'
featureFile = output_dir + '/pool5.npy'
labelFile = output_dir + '/factors.txt'
f = open(labelFile,'r')
labels = []
for l in f:
	labels.append([int(i) for i in l.split()])
f.close()
classLabels = numpy.array(labels)[:,0]
if feature_type=='class':
	labels = numpy.array(labels)[:,0]
elif feature_type=='fgColor':
	labels = [palette[l[2]] for l in labels]
	labels = numpy.array(labels) / 127.5 - 1
elif feature_type=='bgColor':
	labels = [palette[l[3]] for l in labels]
	labels = numpy.array(labels) / 127.5 - 1
elif feature_type=='orientation':
	#labels = [l[1]*numpy.pi/4 for l in labels]
	#labels = numpy.array([[numpy.cos(l),numpy.sin(l)] for l in labels])
	labels = numpy.array(labels)[:,1] 
elif feature_type=='texture':
	labels = numpy.array(labels)[:,2]
data = numpy.load(featureFile)

numpy.random.seed(0)
if feature_type in ['class','texture']:
	solver = caffe.AdamSolver('architecture/solver_class.prototxt')
elif feature_type in ['fgColor','bgColor']:
	solver = caffe.AdamSolver('architecture/solver_color.prototxt')
elif feature_type=='orientation':
	solver = caffe.AdamSolver('architecture/solver_orientation.prototxt')
	classNet = caffe.Net('architecture/pool5_class.prototxt','architecture/class.caffemodel',caffe.TEST)
	solver.net.params['fc1'][0].data[:,:] = classNet.params['fc1'][0].data[:,:]
	solver.net.params['fc1'][1].data[:] = classNet.params['fc1'][1].data[:]
num_samples = data.shape[0]
feature_size = data.shape[1]
batchsize = 50
test_batchsize = int(0.1 * num_samples)
validation_batchsize = min(test_batchsize,100)
solver.net.blobs['data'].reshape(batchsize,feature_size)
solver.test_nets[0].blobs['data'].reshape(validation_batchsize,feature_size)
if feature_type in ['class','texture','orientation']:
	solver.net.blobs['label'].reshape(batchsize)
	solver.test_nets[0].blobs['label'].reshape(validation_batchsize)
else:
	label_size = labels.shape[1]
	solver.net.blobs['label'].reshape(batchsize,label_size)
	solver.test_nets[0].blobs['label'].reshape(validation_batchsize,label_size)

trainIndices = []
testIndices = []
while (len(testIndices) < test_batchsize):
	r = numpy.random.randint(num_samples)
	if not r in testIndices:
		testIndices.append(r)
for i in range(num_samples):
	if not i in testIndices:
		trainIndices.append(i)
numTraining = len(trainIndices)
numTesting = len(testIndices)
for i in range(validation_batchsize):
	solver.test_nets[0].blobs['data'].data[i,:] = data[testIndices[i],:]
	if feature_type in ['class','texture','orientation']:
		solver.test_nets[0].blobs['label'].data[i] = labels[testIndices[i]]
	else:
		solver.test_nets[0].blobs['label'].data[i,:] = labels[testIndices[i],:]
trainIndices = numpy.random.permutation(trainIndices)
		
nepoch = 10
niter = int(nepoch * numTraining / batchsize)
test_interval = 20
train_loss = numpy.zeros(niter)
test_loss = numpy.zeros(niter/test_interval)
print('forward pass '+str(numTraining)+' training '+str(validation_batchsize)+' testing samples')
print('epochs %d iters %d batchsize %d' % (nepoch,niter,batchsize))
start = time.clock()
for it in range(niter):
	for j in range(batchsize):
		id = trainIndices[(it * batchsize + j) % numTraining]
		solver.net.blobs['data'].data[j,:] = data[id,:]
		if feature_type in ['class','texture','orientation']:
			solver.net.blobs['label'].data[j] = labels[id]
		else:
			solver.net.blobs['label'].data[j,:] = labels[id,:]
	solver.step(1)  # SGD by Caffe

	# store the train loss
	train_loss[it] = solver.net.blobs['loss'].data
	if it % test_interval == test_interval - 1:
		test_loss[it/test_interval] = solver.test_nets[0].blobs['loss'].data
		#print 1.0 * numpy.sum(numpy.argmax(solver.test_nets[0].blobs['fc1'].data,axis=1)==classLabels[testIndices]) / len(testIndices)
		# Plots
		plt.subplot(2,1,1)
		plt.plot(range(it), train_loss[0:it], hold = False)
		plt.subplot(2,1,2)
		plt.plot(range(it/test_interval), test_loss[0:it/test_interval], hold = False)
		plt.draw()
		plt.pause(0.01)

	if it > 0 and it % int(numTraining/batchsize) == 0: # Resample Training index per epoch
		trainIndices = numpy.random.permutation(trainIndices)
end = time.clock()
if feature_type in ['class','texture','orientation']:
	pred = numpy.argmax(solver.test_nets[0].blobs['fc3' if feature_type=='orientation' else 'fc1'].data,axis=1)
	correct = numpy.sum(pred == solver.test_nets[0].blobs['label'].data)
	print 'Trained %s with %.2f train loss %.2f test loss (%d / %d acc) (%.2fs / %d iter)' % \
	(feature_type,train_loss[-1],test_loss[-1],correct,validation_batchsize,(end-start),niter)
else:
	print 'Trained %s with %.2f train loss %.2f test loss (%.2fs / %d iter)' % \
	(feature_type,train_loss[-1],test_loss[-1],(end-start),niter)
solver.net.save('architecture/%s.caffemodel' % feature_type)
