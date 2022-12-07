
# check

python3 scripts/sample_diffusion.py \
    -cfg configs/frido/t2i/frido_f16f8_coco_clip.yaml \
    -r exp/t2i/frido_f16f8_coco_clip \
    -G -gs 1.5 -c 200 -name full

fidelity --gpu 0 --fid --input2 exp/t2i/frido_f16f8_coco_clip/samples/full/img/inputs \
    --input1 exp/t2i/frido_f16f8_coco_clip/samples/full/img/sample
