#!/usr/bin/python

import sys

if len(sys.argv) < 2:
	print './cld.py input.pcd out.pgm'
	sys.exit(1)
	
f = open(sys.argv[1],'r')
while True:
	l = f.readline()
	if l.startswith('DATA'):
		break

import numpy
import matplotlib.pyplot as plt

points=[]
while True:
	l = f.readline()
	if not l:
		break
	ll = l.split()
	x = float(ll[0])
	y = float(ll[1])
	z = float(ll[2])
	points.append([x,y,z])
f.close()

def savePGM(out,I):
	f = open(out,'w')
	f.write('P5\n%d %d\n255\n' % (len(I[0]),len(I)))
	for row in I:
		for i in row:
			f.write('%d ' % i)
		f.write('\n')
	f.close()

w=100
h=100
raster=numpy.zeros((h,w),dtype=numpy.uint8)
m = numpy.mean(points,axis=0)
minZ = numpy.min(points,axis=0)[2]
maxZ = numpy.max(points,axis=0)[2]
if maxZ==minZ:
	maxZ=minZ+1
m[2] = (minZ + maxZ) / 2
maxR = 0
for p in points:
	R = numpy.linalg.norm(numpy.array(p) - m)
	if R > maxR:
		maxR = R
for p in points:
	R = numpy.linalg.norm(numpy.array(p) - m)
	theta = numpy.arctan2(p[1]-m[1],p[0]-m[0])
	if theta < 0:
		theta += numpy.pi * 2
	iw = int(theta / numpy.pi / 2 * w)
	ih = int((maxZ-p[2])/(maxZ-minZ) * (h-1))
	raster[ih][iw] = int(R / maxR * 255)
#print raster
plt.imshow(raster)
plt.savefig(sys.argv[2])
#plt.show()
#savePGM(sys.argv[2],raster)
print 'Saved to',sys.argv[2]

