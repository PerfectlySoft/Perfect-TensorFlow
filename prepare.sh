#!/bin/bash
# This script is for CI Server
VERSION=`cat VERSION`
echo 'Clean Temp Files'
rm -rf /tmp/testdata
echo 'Unzip test pack'
tar xzf testpack.tgz -C /tmp/
echo 'download AI model'
curl https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -o /tmp/testdata/in.zip
echo 'unzip model file'
unzip /tmp/testdata/in.zip -d /tmp/testdata/
if  [[ $OSTYPE =~ darwin* ]]; then
	echo 'get darwin library'
	curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-$VERSION.tar.gz -o /tmp/testdata/darwin.lib.tgz
	tar xzf /tmp/testdata/darwin.lib.tgz -C /tmp/testdata
else
	echo 'get linux library'
	curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-$VERSION.tar.gz -o /tmp/testdata/linux.lib.tgz
	tar xzf /tmp/testdata/linux.lib.tgz -C /tmp/testdata
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/testdata/lib
	ldconfig
fi

