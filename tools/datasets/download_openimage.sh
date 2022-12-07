set -e  # exit script if error

echo "Please install wget first!"
echo "Auto set up datasets in \"../datasets\""

echo "Please install AWS Command Line Interface!"
read -r -p "Do you want to install AWS automatically? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    sudo snap install aws-cli --classic
else
    echo "Pass AWS installation."
fi

echo "Create dataset folders and subfolders"
OI_DIR=tmp/openimage  # change this may cause unexpected errors.
mkdir -p $OI_DIR
mkdir -p $OI_DIR/train
mkdir -p $OI_DIR/validation

echo "Download Open Image V6 datasets (train + valid split) in \"./datasets/openimage\"."
read -r -p "The entire OpenImage raw data may takes few days to download. Are you sure to continue? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    mkdir -p $OI_DIR/validation/data
    # aws s3 --no-sign-request sync s3://open-images-dataset/validation $OI_DIR/validation/data
    mkdir -p $OI_DIR/train/data
    # aws s3 --no-sign-request sync s3://open-images-dataset/train $OI_DIR/train/data
else
    echo "Stop downloading Open Image dataset."
    exit 1
fi

echo "Download annotation files."
mkdir -p $OI_DIR/validation/labels
wget -O $OI_DIR/validation/labels/detections.csv https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv
wget -O $OI_DIR/validation/labels/relationships.csv https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-vrd.csv
wget -O $OI_DIR/validation/labels/classifications.csv https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv
wget -O $OI_DIR/validation/labels/segmentations.csv https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv
mkdir -p $OI_DIR/validation/metadata
wget -O $OI_DIR/validation/metadata/attributes.csv https://storage.googleapis.com/openimages/v6/oidv6-attributes-description.csv
wget -O $OI_DIR/validation/metadata/classes.csv https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
wget -O $OI_DIR/validation/metadata/image_ids.csv https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
wget -O $OI_DIR/validation/metadata/segmentation_classes.csv https://storage.googleapis.com/openimages/v5/classes-segmentation.txt
wget -O $OI_DIR/validation/metadata/hierarchy.json https://www.dropbox.com/s/uw6rwow0ka0baz4/hierarchy.json?dl=1

mkdir -p $OI_DIR/train/labels
wget -O $OI_DIR/train/labels/detections.csv https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv
wget -O $OI_DIR/train/labels/relationships.csv https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv
wget -O $OI_DIR/train/labels/classifications.csv https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv
wget -O $OI_DIR/train/labels/segmentations.csv https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv
mkdir -p $OI_DIR/train/metadata
wget -O $OI_DIR/train/metadata/attributes.csv https://storage.googleapis.com/openimages/v6/oidv6-attributes-description.csv
wget -O $OI_DIR/train/metadata/classes.csv https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
wget -O $OI_DIR/train/metadata/image_ids.csv https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv
wget -O $OI_DIR/train/metadata/segmentation_classes.csv https://storage.googleapis.com/openimages/v5/classes-segmentation.txt
wget -O $OI_DIR/train/metadata/hierarchy.json https://www.dropbox.com/s/uw6rwow0ka0baz4/hierarchy.json?dl=1

wget -O $OI_DIR/info.json https://www.dropbox.com/s/zwza1ehlkutgu3q/info.json?dl=1

# echo "Move datasets to \"../datasets\""
# mkdir -p ../datasets
# mv $OI_DIR ../datasets
