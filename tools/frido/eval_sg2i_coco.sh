
# check

python3 scripts/sample_diffusion.py \
    -cfg configs/frido/sg2i/frido_f16f8_coco.yaml \
    -r exp/sg2i/frido_f16f8_coco \
    -G -gs 1.5 -c 200 -name full

fidelity --gpu 0 --fid --input2 exp/sg2i/frido_f16f8_coco/samples/full/img/inputs \
    --input1 exp/sg2i/frido_f16f8_coco/samples/full/img/sample
