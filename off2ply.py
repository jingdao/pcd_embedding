#!/usr/bin/python

import sys
if len(sys.argv) < 3:
	print './off2ply.py in.off out.ply'
	sys.exit(1)

flipYZ=False
f = open(sys.argv[1],'r')
g = open(sys.argv[2],'w')
l = f.readline()
l = f.readline()
numVertex = int(l.split()[0])
numFace = int(l.split()[1])
header="""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
element face %d
property list uchar int vertex_indices
end_header
""" % (numVertex,numFace)
g.write(header)
if flipYZ:
	for i in range(numVertex):
		l = f.readline().split()
		x = float(l[0])
		y = -float(l[2])
		z = float(l[1])
		g.write("%f %f %f\n"%(x,y,z))
else:
	for i in range(numVertex):
		g.write(f.readline())
for j in range(numFace):
	g.write(f.readline())
f.close()
g.close()
