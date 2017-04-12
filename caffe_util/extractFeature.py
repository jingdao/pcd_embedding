import numpy
import matplotlib.pyplot as plt
import scipy
import time
import os
os.environ['GLOG_minloglevel']="1"
import caffe
caffe.set_mode_gpu()

caffe_root = 'C:/Users/jchen490/Downloads/caffe-windows/'
mu = numpy.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')	
mu = mu.mean(1).mean(1)
model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
labels_file = 'synset_words.txt'
output_dir = 'E:/jd/Documents/pcd_embedding/ModelNet10_large'
labels=[]
for line in open(labels_file,'r'):
	labels.append(line[10:].strip())

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
net.blobs['data'].reshape(1,3,227,227)
	
print 'Loaded caffe net'

num_images = 24000
'''
images = []
for i in range(num_images):
	print i
	I=scipy.misc.imread('%s/%d.ppm' % (output_dir,i))
	I=scipy.misc.imresize(I,(227,227,3))
	I=numpy.array([I[:,:,2] - mu[0],I[:,:,1] - mu[1],I[:,:,0] - mu[2]])
	images.append(I)
	
print 'Loaded %d images' % len(images)
'''
	 
start = time.clock()
#fc7=numpy.zeros((num_images,4096))
#fc6=numpy.zeros((num_images,4096))
pool5=numpy.zeros((num_images,9216),dtype=numpy.float32)
for i in range(num_images):
	print i
	I=scipy.misc.imread('%s/%d.ppm' % (output_dir,i))
	I=scipy.misc.imresize(I,(227,227,3))
	I=numpy.array([I[:,:,2] - mu[0],I[:,:,1] - mu[1],I[:,:,0] - mu[2]])
	net.blobs['data'].data[0,:,:,:] = I
	output = net.forward()
	#fc7[i,:] = net.blobs['fc7'].data[0]
	#fc6[i,:] = net.blobs['fc6'].data[0]
	pool5[i,:] = numpy.resize(net.blobs['pool5'].data[0],9216)
end = time.clock()
avgtime = (end - start) / num_images
print 'Average time: %.3fs' % avgtime

#numpy.save('%s/fc7.npy' % output_dir,fc7)
#numpy.save('%s/fc6.npy' % output_dir,fc6)
numpy.save('%s/pool5.npy' % output_dir,pool5)
