#!/bin/bash
OSABR=$(echo $(uname)|tr '[:upper:]' '[:lower:]')
DWN=/tmp/libtensorflow.tgz
VERSION=`cat VERSION`
URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$OSABR-x86_64-$VERSION.tar.gz
echo $URL
curl $URL -o $DWN
echo "installing binaries ..."
tar xf $DWN -C /usr/local ./lib/libtensorflow.so ./lib/libtensorflow_framework.so
rm -f $DWN