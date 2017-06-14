OSABR=$(echo $(uname)|tr '[:upper:]' '[:lower:]')
DWN=/tmp/libtensorflow.tgz
DIR=/tmp/libtf
rm -rf $DIR
mkdir $DIR
rm -f $DWN
URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$OSABR-x86_64-1.1.0.tar.gz
echo $URL
curl $URL -o $DWN
tar xzf $DWN -C $DIR
sudo cp $DIR/lib/libtensorflow.so /usr/local/lib
rm -rf $DIR
rm -f $DWN
ls -l /usr/local/lib/libtensorflow.so
