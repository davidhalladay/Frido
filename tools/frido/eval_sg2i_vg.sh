
# check

python3 scripts/sample_diffusion.py \
    -cfg configs/frido/sg2i/frido_f16f8_vg.yaml \
    -r exp_check/sg2i/frido_f16f8_vg \
    -G -gs 1.5 -c 200 -name full2

# fidelity --gpu 0 --fid --input2 exp/sg2i/frido_f16f8_vg/samples/full/img/inputs \
#     --input1 exp/sg2i/frido_f16f8_vg/samples/full/img/sample

