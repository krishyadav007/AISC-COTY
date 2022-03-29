mkdir models
cd models
mkdir research
cd research
mkdir deploy
cd ..
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .