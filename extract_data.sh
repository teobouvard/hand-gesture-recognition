#!/bin/bash

mkdir -p data

if [ -f "$1" ] && unzip -t $1; then
	unzip $1 -d /tmp/dataset
else
	echo "$1 is not a zip file"
	exit
fi

for i in {0..5}
do
	mkdir -p data/$i
	/bin/cp /tmp/dataset/train/*_$i*.png data/$i
done

rm -r /tmp/dataset
