#!/bin/bash
OSABR=$(echo $(uname)|tr '[:upper:]' '[:lower:]')
DWN=/tmp/libtensorflow.tgz
VERSION=`cat VERSION`
URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$OSABR-x86_64-$VERSION.tar.gz
wget $URL -O $DWN
sudo tar xf $DWN -C /usr/local ./lib/libtensorflow.so ./lib/libtensorflow_framework.so
rm -f $DWN