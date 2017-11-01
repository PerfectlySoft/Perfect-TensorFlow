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

