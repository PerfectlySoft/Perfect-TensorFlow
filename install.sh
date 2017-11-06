#!/bin/bash
OSABR=$(echo $(uname)|tr '[:upper:]' '[:lower:]')
DWN=/tmp/libtensorflow.tgz
if [ -z "$1" ]
then
	LOC=/usr/local
else
	LOC=$1
fi
VERSION=`cat VERSION`
URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$OSABR-x86_64-$VERSION.tar.gz
echo $URL
curl $URL -o $DWN
echo "installing binaries ..."
tar xf $DWN -C $LOC ./lib/libtensorflow.so ./lib/libtensorflow_framework.so
touch $LOC/lib/libtensorflow.so
touch $LOC/lib/libtensorflow_framework.so
rm -f $DWN