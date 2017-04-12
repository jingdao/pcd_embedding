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
'''
'orientation':[
	[1,0],
	[0.707,0.707],
	[0,1],
	[-0.707,0.707],
	[-1,0],
	[-0.707,-0.707],
	[0,-1],
	[0.707,-0.707],
],
'''

regression_names = {
	'fgColor':['black','white','red','green','blue','yellow','purple','cyan','orange','violet'],
	'bgColor':['black','white','red','green','blue','yellow','purple','cyan','orange','violet'],
	#'orientation':['N','NE','E','SE','S','SW','W','NW']
}
classification_names = {
	'class':['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet'],
	'texture':['porcelain','checkered','dark_wood','light_wood','bamboo','striped','tufted','cloth','metallic','flowery'],
	'orientation':['horizontal','vertical','diagonal']
}

alexnet = caffe.Net(model_def,model_weights,caffe.TEST)
alexnet.blobs['data'].reshape(1,3,227,227)
print 'Loaded AlexNet'
	
images = []
image_names = []
image_dir = '101_ObjectCategories/test'
#image_list = ['image_%04d.jpg' % i for i in range(1,63)]
#image_dir = 'E:/jd/Documents/pcd_embedding/ModelNet10_large'
image_list = [str(i)+'.ppm' for i in range(60)]
for i in image_list:
	if i.endswith('.jpg') or i.endswith('.ppm'):
		I=scipy.misc.imread(image_dir + '/' + i)
		if len(I.shape) == 2:
			I=scipy.misc.imresize(I,(227,227))
			I=numpy.array([I[:,:] - mu[0],I[:,:] - mu[1],I[:,:] - mu[2]])
		else:
			I=scipy.misc.imresize(I,(227,227,3))
			I=numpy.array([I[:,:,2] - mu[0],I[:,:,1] - mu[1],I[:,:,0] - mu[2]])
		images.append(I)
		image_names.append(i)
	
print 'Loaded %d images' % len(images)

nets = {
	'class': caffe.Net('architecture/pool5_class.prototxt','architecture/class.caffemodel',caffe.TEST),
	'orientation': caffe.Net('architecture/pool5_orientation.prototxt','architecture/orientation.caffemodel',caffe.TEST),
	'texture': caffe.Net('architecture/pool5_class.prototxt','architecture/texture.caffemodel',caffe.TEST),
	'fgColor': caffe.Net('architecture/pool5_color.prototxt','architecture/fgColor.caffemodel',caffe.TEST),
	'bgColor': caffe.Net('architecture/pool5_color.prototxt','architecture/bgColor.caffemodel',caffe.TEST)
}
for k in nets:
	nets[k].blobs['data'].reshape(1,9216)
	if k in ['class','texture','orientation']:
		nets[k].blobs['label'].reshape(1)
	#elif k == 'orientation':
		#nets[k].blobs['label'].reshape(1,2)
	else:
		nets[k].blobs['label'].reshape(1,3)
nbrs={
	'fgColor': NearestNeighbors(n_neighbors=1, algorithm='brute').fit(numpy.array(regression_values['fgColor'])/255.0),
	'bgColor': NearestNeighbors(n_neighbors=1, algorithm='brute').fit(numpy.array(regression_values['bgColor'])/255.0),
	#'orientation': NearestNeighbors(n_neighbors=1, algorithm='brute').fit(regression_values['orientation']),
}

print 'Loaded finetuned nets'

features={k:[] for k in nets.keys()}
features['_pool5']=[]
start = time.clock()
for i in range(len(images)):
	d = images[i]
	alexnet.blobs['data'].data[0,:,:,:] = d
	output = alexnet.forward()
	pool5 = numpy.resize(alexnet.blobs['pool5'].data[0],9216)
	features['_pool5'].append(pool5)
	s = image_names[i]
	for k in sorted(nets.keys()):
		nets[k].blobs['data'].data[0,:] = pool5
		nets[k].forward()
		if k == 'orientation':
			f = numpy.array(nets[k].blobs['fc3'].data[0,:])
		else:
			f = numpy.array(nets[k].blobs['fc1'].data[0,:])
		features[k].append(f)
		if k in ['class','texture','orientation']:
			s += ' ' + classification_names[k][numpy.argmax(f)]
			#s += ' ' + str(numpy.argmax(f))
		else:
			id = nbrs[k].kneighbors([f])[1][0,0]
			s += ' ' + regression_names[k][id]
			#s += ' ' + str(id)
	print s
end = time.clock()
avgtime = (end - start) / len(images)
print 'Average time: %.3fs' % avgtime

indices  = [[i for i in range(len(images))]]
signature = numpy.zeros((len(images),29),dtype=numpy.float32)
j = 0
for k in sorted(features.keys()):
	if not k=='_pool5':
		D = len(features[k][0])
		signature[:,j:j+D] = features[k]
		j += D
	n = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(features[k])
	match = n.kneighbors(features[k])[1][:,1]
	#match = [image_names[m] for m in match]
	indices.append(match)
indices = zip(*indices)
#for i in indices:
	#print indices
numpy.save('%s/signature.npy' % image_dir,signature)	

num_samples = 5
template = numpy.zeros((170*num_samples,len(indices[0])*100+20,3),dtype=numpy.uint8)
def get_image(i):
	I=scipy.misc.imread(image_dir + '/' + image_names[i])
	if len(I.shape) == 2:
		I=scipy.misc.imresize(I,(150,100))
		I=numpy.dstack((I,I,I))
	else:
		I=scipy.misc.imresize(I,(150,100,3))
	return I
print sorted(features.keys())
for i in range(num_samples):
	print indices[i]
	template[170*i:170*i+150, :100, :] = get_image(indices[i][0])
	for j in range(1,len(indices[i])):
		template[170*i:170*i+150, 20+j*100:20+(j+1)*100, :] = get_image(indices[i][j])
scipy.misc.imsave('%s/template.jpg' % image_dir, template)
	
