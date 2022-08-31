set -e  # exit script if error

echo "Please install wget first!"
echo "Auto set up datasets in \"../datasets\""

echo "Create dataset folders and subfolders"
mkdir datasets
mkdir datasets/coco
mkdir datasets/coco/2014
mkdir datasets/coco/2017

echo "Download coco 2014 datasets (valid split) in \"./datasets/coco/2014\"."
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

mv val2014.zip datasets/coco/2014
mv annotations_trainval2014.zip datasets/coco/2014
unzip datasets/coco/2014/val2014.zip -d datasets/coco/2014
unzip datasets/coco/2014/annotations_trainval2014.zip -d datasets/coco/2014

echo "Download coco 2017 datasets (valid split) in \"./datasets/coco/2017\"."
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

mv val2017.zip datasets/coco/2017
mv annotations_trainval2017.zip datasets/coco/2017
mv stuff_annotations_trainval2017.zip datasets/coco/2017
unzip datasets/coco/2017/val2017.zip -d datasets/coco/2017
unzip datasets/coco/2017/annotations_trainval2017.zip -d datasets/coco/2017
unzip datasets/coco/2017/stuff_annotations_trainval2017.zip -d datasets/coco/2017

echo "Remove cache files..."
rm -rf datasets/coco/2014/val2014.zip
rm -rf datasets/coco/2014/annotations_trainval2014.zip
rm -rf datasets/coco/2017/val2017.zip
rm -rf datasets/coco/2017/annotations_trainval2017.zip
rm -rf datasets/coco/2017/stuff_annotations_trainval2017.zip

echo "Move datasets to \"../datasets\""
mv datasets ../
