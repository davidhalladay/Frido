

# Note that most parts of this script comes from the official codebase of Sg2Im.
# Please refer to thier code: https://github.com/google/sg2im
# If you find this useful, please also cite their paper.

echo "Please install wget first!"
echo "Auto set up datasets in \"../datasets\""

# download vg
echo "Create dataset folders and subfolders"
VG_DIR=tmp/vg  # change this may cause unexpected errors.
mkdir -p $VG_DIR

echo "Download Visual Genome datasets (train + valid split) in \"$VG_DIR\"."
wget https://visualgenome.org/static/data/dataset/objects.json.zip -O $VG_DIR/objects.json.zip
wget https://visualgenome.org/static/data/dataset/attributes.json.zip -O $VG_DIR/attributes.json.zip
wget https://visualgenome.org/static/data/dataset/relationships.json.zip -O $VG_DIR/relationships.json.zip
wget https://visualgenome.org/static/data/dataset/object_alias.txt -O $VG_DIR/object_alias.txt
wget https://visualgenome.org/static/data/dataset/relationship_alias.txt -O $VG_DIR/relationship_alias.txt
wget https://visualgenome.org/static/data/dataset/image_data.json.zip -O $VG_DIR/image_data.json.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O $VG_DIR/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O $VG_DIR/images2.zip

# download coco annotation (for pre-process only)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $VG_DIR/annotations_trainval2017.zip

unzip $VG_DIR/objects.json.zip -d $VG_DIR
unzip $VG_DIR/attributes.json.zip -d $VG_DIR
unzip $VG_DIR/relationships.json.zip -d $VG_DIR
unzip $VG_DIR/image_data.json.zip -d $VG_DIR
unzip $VG_DIR/images.zip -d $VG_DIR/images
unzip $VG_DIR/images2.zip -d $VG_DIR/images2
unzip $VG_DIR/annotations_trainval2017.zip -d $VG_DIR

echo "Start pre-process data by using the scripts provided from Sg2Im."
mkdir -p $VG_DIR/VG_100K
mv $VG_DIR/images/VG_100K/* $VG_DIR/VG_100K
mv $VG_DIR/images2/VG_100K_2/* $VG_DIR/VG_100K
git clone https://github.com/google/sg2im.git 
mv sg2im $VG_DIR

# pre-processing
python3 scripts/preprocess_vg_sg2im.py

# convert into coco annotation format
echo "Convert h5py files into coco annotations json format."
python3 scripts/convert_vg_to_coco_style.py -b $VG_DIR -s val
python3 scripts/convert_vg_to_coco_style.py -b $VG_DIR -s train

# Generate files for scene-graph-to-image task 
echo "Create scene-graph-to-image annotations."
python3 scripts/preprocess_vg_to_sg.py -b $VG_DIR -s val
python3 scripts/preprocess_vg_to_sg.py -b $VG_DIR -s train

# clean zip
echo "Remove cache files..."
rm -rf $VG_DIR/images
rm -rf $VG_DIR/images2
rm -rf $VG_DIR/annotations
rm -rf $VG_DIR/sg2im
rm -rf $VG_DIR/*.zip
rm -rf $VG_DIR/*.h5

echo "Move datasets to \"../datasets\""
mkdir -p ../datasets
mv $VG_DIR ../datasets