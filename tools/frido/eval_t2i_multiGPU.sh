set -e 
# We demo how to run Frido inference in multi-GPUs.

# ngpu: number of gpus
# igpu: idx of the gpu for each command  

NUMGPU=$(expr $1 - 1)

for i in $(seq 0 $NUMGPU); do \
    echo Running on GPU $i 
    CUDA_VISIBLE_DEVICES=$i python3 scripts/sample_diffusion.py -cfg configs/frido/t2i/frido_f16f8_coco.yaml -r exp/t2i/frido_f16f8_coco -G -gs 1.5 -c 200 -ngpu $1 -igpu $i -name full & \
done; wait