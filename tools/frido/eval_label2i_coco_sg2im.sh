
# check

python3 scripts/sample_diffusion.py \
    -cfg configs/frido/label2i/frido_f16f8_coco_sg2im.yaml \
    -r exp/label2i/frido_f16f8_coco_sg2im \
    -c 200 -name full

fidelity --gpu 0 --fid --input2 exp/label2i/frido_f16f8_coco_sg2im/samples/full/img/inputs \
    --input1 exp/label2i/frido_f16f8_coco_sg2im/samples/full/img/sample
