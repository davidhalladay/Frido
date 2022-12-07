set -e  # exit script if error

echo "Please install wget first!"
echo "Auto set up datasets in \"../datasets\""

echo "Create dataset folders and subfolders"
mkdir datasets
mkdir datasets/coco
mkdir datasets/coco/2014
mkdir datasets/coco/2017

echo "Download coco 2014 datasets (train + valid split) in \"./datasets/coco/2014\"."
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

mv train2014.zip datasets/coco/2014
mv val2014.zip datasets/coco/2014
mv annotations_trainval2014.zip datasets/coco/2014
unzip datasets/coco/2014/train2014.zip -d datasets/coco/2014
unzip datasets/coco/2014/val2014.zip -d datasets/coco/2014
unzip datasets/coco/2014/annotations_trainval2014.zip -d datasets/coco/2014

echo "Download coco 2017 datasets (train + valid split) in \"./datasets/coco/2017\"."
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

mv train2017.zip datasets/coco/2017
mv val2017.zip datasets/coco/2017
mv annotations_trainval2017.zip datasets/coco/2017
mv stuff_annotations_trainval2017.zip datasets/coco/2017
unzip datasets/coco/2017/train2017.zip -d datasets/coco/2017
unzip datasets/coco/2017/val2017.zip -d datasets/coco/2017
unzip datasets/coco/2017/annotations_trainval2017.zip -d datasets/coco/2017
unzip datasets/coco/2017/stuff_annotations_trainval2017.zip -d datasets/coco/2017

wget -O datasets/coco/2017/annotations/scenegraph_train2017.json https://www.dropbox.com/s/89p4ynvvt2cn06e/scenegraph_train2017.json?dl=1
wget -O datasets/coco/2017/annotations/scenegraph_val2017.json https://www.dropbox.com/s/6fn60314c91zte8/scenegraph_val2017.json?dl=1

echo "Remove cache files..."
rm -rf datasets/coco/2014/train2014.zip
rm -rf datasets/coco/2014/val2014.zip
rm -rf datasets/coco/2014/annotations_trainval2014.zip
rm -rf datasets/coco/2017/train2017.zip
rm -rf datasets/coco/2017/val2017.zip
rm -rf datasets/coco/2017/annotations_trainval2017.zip
rm -rf datasets/coco/2017/stuff_annotations_trainval2017.zip

echo "Move datasets to \"../datasets\""
mv datasets ../
