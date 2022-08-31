set -e  # exit script if error

BLOB_URL='https://acvrpublicycchen.blob.core.windows.net/frido'

install_azcopy () {
    # install azcopy
    # url can be found by
    # curl -i https://aka.ms/downloadazcopy-v10-linux | grep Location
    wget https://azcopyvnext.azureedge.net/release20220721/azcopy_linux_amd64_10.16.0.tar.gz &&\
    tar -zxvf azcopy_linux_amd64_10.16.0.tar.gz &&\
    mv azcopy_linux_amd64_10.16.0/azcopy ./ &&\
    chmod +x ./azcopy &&\
    rm -r azcopy_linux_amd64_10.16.0*
}

echo "check azcopy install."
if ./azcopy -v; then
    echo "azcopy is installed"
else
    echo "azcopy not found, installing it now..."
    install_azcopy
fi

# create dirs
for DIR in 'exp/msvqgan/openimage_f16f8' \
           'exp/msvqgan/openimage_f8f4' \
	   'exp/t2i/frido_f16f8/checkpoints' \
	   'exp/layout2i/frido_f8f4/checkpoints'; do
    if [ ! -d $DIR ] ; then
        mkdir -p $DIR
    fi
done

# download msvqgan model from Azure blob
echo "Start downloading MS-VQGAN weights..."
./azcopy copy "${BLOB_URL}/public_models/msvqgan/openimage_f16f8/model.ckpt" ./

echo "Download successfully."
echo "Auto move model weights to corresponding folder."
mv model.ckpt exp/msvqgan/openimage_f16f8/model.ckpt

echo "Start downloading MS-VQGAN weights..."
./azcopy copy "${BLOB_URL}/public_models/msvqgan/openimage_f8f4/model.ckpt" ./
echo "Download successfully."
echo "Auto move model weights to corresponding folder."
mv model.ckpt exp/msvqgan/openimage_f8f4/model.ckpt

# download t2i model from Azure blob
echo "Start downloading Frido weights for t2i task..."
./azcopy copy "${BLOB_URL}/public_models/t2i/model.ckpt" ./
echo "Download successfully."
echo "Auto move model weights to corresponding folder."
mv model.ckpt exp/t2i/frido_f16f8/checkpoints/model.ckpt

# download layout2i model from Azure blob
echo "Start downloading Frido weights for layout2i task..."
./azcopy copy "${BLOB_URL}/public_models/layout2i/model.ckpt" ./
echo "Download successfully."
echo "Auto move model weights to corresponding folder."
mv model.ckpt exp/layout2i/frido_f8f4/checkpoints/model.ckpt

