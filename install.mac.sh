cd /tmp
rm -rf tf
mkdir tf
cd tf
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.1.0.tar.gz
tar xzvf ./libtensorflow-cpu-darwin-x86_64-1.1.0.tar.gz
sudo cp ./lib/libtensorflow.so /usr/local/lib
cd /tmp
rm -rf tf
cd
