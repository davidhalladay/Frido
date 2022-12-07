
# check

python3 scripts/sample_diffusion.py \
    -cfg configs/frido/t2i/frido_f16f8_coco.yaml \
    -r exp/t2i/frido_f16f8_coco \
    -e 0 -G -gs 1.5 -c 50 -plms -name full

fidelity --gpu 0 --fid --input2 exp/t2i/frido_f16f8_coco/samples/full/img/inputs \
    --input1 exp/t2i/frido_f16f8_coco/samples/full/img/sample
