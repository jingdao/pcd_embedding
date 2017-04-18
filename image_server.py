import numpy
import scipy
import time
import os
import sys
from sklearn.neighbors import NearestNeighbors

use_caffe = False

if use_caffe:
	os.environ['GLOG_minloglevel']="1"
	import caffe
	caffe.set_mode_gpu()

	caffe_root = 'C:/Users/jchen490/Downloads/caffe-windows/'
	mu = numpy.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')	
	mu = mu.mean(1).mean(1)
	model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
	model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'

regression_values = {
	'fgColor':[
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
	],
	'bgColor':[
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
	],
}

if use_caffe:
	alexnet = caffe.Net(model_def,model_weights,caffe.TEST)
	alexnet.blobs['data'].reshape(1,3,227,227)
	print 'Loaded AlexNet'

	nets = {
		'class': caffe.Net('caffe_util/architecture/pool5_class.prototxt','caffe_util/architecture/class.caffemodel',caffe.TEST),
		'orientation': caffe.Net('caffe_util/architecture/pool5_orientation.prototxt','caffe_util/architecture/orientation.caffemodel',caffe.TEST),
		'texture': caffe.Net('caffe_util/architecture/pool5_class.prototxt','caffe_util/architecture/texture.caffemodel',caffe.TEST),
		'fgColor': caffe.Net('caffe_util/architecture/pool5_color.prototxt','caffe_util/architecture/fgColor.caffemodel',caffe.TEST),
		'bgColor': caffe.Net('caffe_util/architecture/pool5_color.prototxt','caffe_util/architecture/bgColor.caffemodel',caffe.TEST)
	}
	for k in nets:
		nets[k].blobs['data'].reshape(1,9216)
		if k in ['class','texture','orientation']:
			nets[k].blobs['label'].reshape(1)
		#elif k == 'orientation':
			#nets[k].blobs['label'].reshape(1,2)
		else:
			nets[k].blobs['label'].reshape(1,3)
	print 'Loaded finetuned nets'

#image_dir = 'caffe_util/101_ObjectCategories/chair'
#image_list = ['image_%04d.jpg' % i for i in range(1,63)]
#image_dir = 'caffe_util/101_ObjectCategories/imagenet'
#image_list = ['%d.jpg' % i for i in range(0,2465)]
image_dir = 'caffe_util/101_ObjectCategories/test'
image_list = ['%d.jpg' % i for i in range(0,2400)]
signature = numpy.load(image_dir+'/signature.npy')

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()
	#return x

if use_caffe:
	def get_signature(image_path):
		I=scipy.misc.imread(image_path)
		if len(I.shape) == 2:
			I=scipy.misc.imresize(I,(227,227))
			I=numpy.array([I[:,:] - mu[0],I[:,:] - mu[1],I[:,:] - mu[2]])
		else:
			I=scipy.misc.imresize(I,(227,227,3))
			I=numpy.array([I[:,:,2] - mu[0],I[:,:,1] - mu[1],I[:,:,0] - mu[2]])
		alexnet.blobs['data'].data[0,:,:,:] = I
		output = alexnet.forward()
		pool5 = numpy.resize(alexnet.blobs['pool5'].data[0],9216)
		S = numpy.zeros(29,dtype=numpy.float32)
		j = 0
		for k in sorted(nets.keys()):
			nets[k].blobs['data'].data[0,:] = pool5
			nets[k].forward()
			if k == 'orientation':
				f = numpy.array(nets[k].blobs['fc3'].data[0,:])
			else:
				f = numpy.array(nets[k].blobs['fc1'].data[0,:])
			D = len(f)
			S[j:j+D] = f
			j += D
		return S
	
def get_path_from_id(image_id):
	try:
		image_id = int(image_id)
		return image_dir + '/' + image_list[image_id]
	except (ValueError,IndexError,TypeError):
		return 'default.jpg'

def get_retrieval_results(image_path,query):
	if use_caffe:
		S = get_signature(image_path)
	else:
		S = signature[int(query['id'][0])]
	err = signature - S
	k = query['retrieve'][0]
	if k=='bgColor':
		err = err[:,0:3]
	elif k=='class':
		err = err[:,3:13]
	elif k=='fgColor':
		err = err[:,13:16]
	elif k=='orientation':
		err = numpy.hstack((err[:,3:13],err[:,16:19]))
	elif k=='texture':
		err = err[:,19:29]
	err = err**2
	err = numpy.sum(err,axis=1)
	top = numpy.argsort(err)[1:10]
	return [image_dir+'/'+image_list[i] for i in top]
		
def get_query_results(image_path,query):
	if use_caffe:
		S = get_signature(image_path)
	else:
		S = signature[int(query['id'][0])]
	
	valid = set(range(len(signature)))
	weight = numpy.ones(29)
	for k in query:
		i = int(query[k][0])
		if i==0:
			continue
		if k=='bgColor':
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(numpy.array(regression_values[k]) / 127.5 - 1)
			dist, index = nbrs.kneighbors(signature[:,0:3])
			index = numpy.array([j[0]==(i-1) for j in index])
			v = set(numpy.nonzero(index)[0])
			valid = valid & v
			weight[0:3] = 0
		elif k=='class':
			match = numpy.argmax(signature[:,3:13],axis=1) == (i-1)
			v = set(numpy.nonzero(match)[0])
			valid = valid & v
			weight[3:13] = 0
		elif k=='fgColor':
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(numpy.array(regression_values[k]) / 127.5 - 1)
			dist, index = nbrs.kneighbors(signature[:,13:16])
			index = numpy.array([j[0]==(i-1) for j in index])
			v = set(numpy.nonzero(index)[0])
			valid = valid & v
			weight[13:16] = 0
		elif k=='orientation':
			match = numpy.argmax(signature[:,16:19],axis=1) == (i-1)
			v = set(numpy.nonzero(match)[0])
			valid = valid & v
			weight[16:19] = 0
		elif k=='texture':
			match = numpy.argmax(signature[:,19:29],axis=1) == (i-1)
			v = set(numpy.nonzero(match)[0])
			valid = valid & v
			weight[19:29] = 0
	
	err = signature - S
	err = err**2
	err = err * weight
	err = numpy.sum(err,axis=1)
	order = numpy.argsort(err)
	order = filter(lambda x:x in valid, order)
	top = order[1:10]
	return [image_dir+'/'+image_list[i] for i in top]
	
	





