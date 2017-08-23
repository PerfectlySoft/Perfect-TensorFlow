OSABR=$(echo $(uname)|tr '[:upper:]' '[:lower:]')
DWN=/tmp/libtensorflow.tgz
VERSION=1.3.0
URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$OSABR-x86_64-$VERSION.tar.gz
echo $URL
wget $URL -O $DWN
sudo tar xvf $DWN -C /usr/local ./lib/libtensorflow.so
rm -f $DWN
ls -l /usr/local/lib/libtensorflow.so
