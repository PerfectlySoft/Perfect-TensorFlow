# This script is for CI Server
rm -rf /tmp/testdata
tar xzf testpack.tgz -C /tmp/
wget -q https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.1.0.tar.gz -O /tmp/testdata/linux.lib.tgz
mkdir /tmp/testdata/linux
tar xzf /tmp/testdata/linux.lib.tgz -C /tmp/testdata/linux
wget -q https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.1.0.tar.gz -O /tmp/testdata/darwin.lib.tgz
mkdir /tmp/testdata/darwin
tar xzf /tmp/testdata/darwin.lib.tgz -C /tmp/testdata/darwin
wget -q https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -O /tmp/testdata/in.zip
unzip /tmp/testdata/in.zip -d /tmp/testdata/
swift test > test-results.txt
# swift test
# rm -rf /tmp/testdata
