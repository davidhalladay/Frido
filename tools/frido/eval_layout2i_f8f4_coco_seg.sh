
# check

python3 scripts/sample_diffusion.py \
    -cfg configs/frido/layout2i/frido_f8f4_coco_seg.yaml \
    -r exp/layout2i/frido_f8f4_coco_seg \
    -c 200 -name full


fidelity --gpu 0 --fid --input2 exp/layout2i/frido_f8f4_coco_seg/samples/full/img/inputs \
    --input1 exp/layout2i/frido_f8f4_coco_seg/samples/full/img/sample
