#!/bin/bash

models=/media/jd/9638A1E538A1C519/Users/jchen490/Desktop/3DShapeNetsCode/3DShapeNets/ModelNet10
dir=ModelNet10
numSamples=10
classID=0
sampleID=0

getPLY=false
getProjection=false
getFeature=false
renderPLY=true

if $getPLY
then
	rm -f $dir/labels.txt
	for f in $models/*
	do
		if [ -d $f ]
		then
			class=`basename $f`
			for i in `seq 1 $numSamples`
			do
				src=$models/$class/train/$class"_"`printf %04d.off $i`
				dst=`printf $dir/%d.ply $sampleID`
				./off2ply.py $src $dst
				echo `printf "%d %s" $classID $class` >> $dir/labels.txt
				((sampleID++))
			done
			((classID++))
		fi
	done
fi

if $getProjection
then
	numModels=`cat $dir/labels.txt | wc -l`
	((numModels--))
	rm -f $dir/*.pcd
	for i in `seq 0 $numModels`
	do
		src=`printf $dir/%d.ply $i`
		~/Documents/PointCloudApp/zbuffer $src $dir
	done
fi

if $getFeature
then
	numModels=`ls $dir/*.pcd | wc -l`
	((numModels--))
	for i in `seq 0 $numModels`
	do
		src=`printf $dir/%d-cloud.pcd $i`
		dst=`printf $dir/%d.png $i`
		./cld.py $src $dst
	done
fi

if $renderPLY
then
	rm -f $dir/*.ppm
	rm -f $dir/factors.txt
	numModels=`cat $dir/labels.txt | wc -l`
	((numModels--))
	cd $dir
	for i in `seq 0 $numModels`
	do
		src=`printf %d.ply $i`
		~/Documents/PointCloudApp/renderPLY $src ./
	done
	cd ..
fi
