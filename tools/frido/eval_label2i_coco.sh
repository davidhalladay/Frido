
# check

python3 scripts/sample_diffusion.py \
    -cfg configs/frido/label2i/frido_f16f8_coco.yaml \
    -r exp/label2i/frido_f16f8_coco \
    -c 200 -name full

fidelity --gpu 0 --fid --input2 exp/label2i/frido_f16f8_coco/samples/full/img/inputs \
    --input1 exp/label2i/frido_f16f8_coco/samples/full/img/sample
