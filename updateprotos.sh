PROJ_BASE=$PWD
TF=/tmp/tf
PTF=/tmp/ptf
OUTP=/tmp/pbout
rm -rf $TF
rm -rf $PTF
rm -rf $OUTP
mkdir $OUTP
if  [[ $OSTYPE =~ darwin* ]]; then
  brew install protobuf
else
  sudo apt-get install protobuf-compiler
fi
git clone https://github.com/tensorflow/tensorflow.git $TF
swift build -c release --build-path $PTF
find $TF -type f -iname "*.proto" -print0 | while IFS= read -r -d $'\0' LINE; do
  protoc --plugin=$PTF/release/protoc-gen-swift --swift_opt=Visibility=Public --swift_out=$OUTP --proto_path=$TF $LINE
done
find $OUTP -type f -iname "*.pb.swift" -print0 | while IFS= read -r -d $'\0' LINE; do
  FILE=$(basename $LINE)
  DEST=$(sed 's#.pb.swift#.swift#g' <<< $FILE)
  cp $LINE Sources/PerfectTensorFlow/pb.$DEST
done
