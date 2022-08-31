# *Frido*: Feature Pyramid Diffusion for Complex Scene Image Synthesis

---
This is the official repository of [Frido](https://arxiv.org/abs/2208.13753). Currently, text-to-image and layout-to-image for COCO are supported (inference-only). We will release more pre-trained models for other image synthesis tasks. Training code will be available in future version. Please stay tuned!

![Frido](./figures/model.png)
![demo](./figures/demo.png)

---
## Machine environment
- Ubuntu version: 18.04.5 LTS
- CUDA version: 11.6
- Testing GPU: Nvidia Tesla V100
---

## Requirements
A [conda](https://conda.io/) environment named `frido` can be created and activated with:

```bash
conda env create -f environment.yaml
conda activate frido
```
---

## Datasets setup
We provide two approaches to set up the datasets:
### Auto-download
To automatically download datasets and save it into the default path (`../`), please use following script:
```bash
bash tools/download_datasets.sh
```
### Manual setup

#### Text-to-image generation
- We use COCO 2014 splits for text-to-image task, which can be downloaded from [official COCO website](https://cocodataset.org/#download).
- Please create a folder name `2014` and collect the downloaded data and annotations as follows.

   <details><summary>COCO 2014 file structure</summary>

   ```
   >2014
   ├── annotations
   │   └── captions_val2014.json
   │   └── ...
   └── val2014
      └── COCO_val2014_000000000073.jpg
      └── ... 
   ```

   </details>


#### Layout-to-image generation
- We use COCO 2017 splits to test Frido on layout-to-image task, which can be downloaded from [official COCO website](https://cocodataset.org/#download).
- Please create a folder name `2017` and collect the downloaded data and annotations as follows.

   <details><summary>COCO 2017 file structure</summary>

   ```
   >2017
   ├── annotations
   │   └── captions_val2017.json
   │   └── ...
   └── val2017
      └── 000000000872.jpg
      └── ... 
   ```

   </details>


#### File structure for dataset and code
Please make sure that the file structure is the same as the following. Or, you might modify the config file to match the corresponding paths.

   <details><summary>File structure</summary>

   ```
   >datasets
   ├── coco
   │   └── 2014
   │        └── annotations
   │        └── val2014
   │        └── ...
   │   └── 2017
   │        └── annotations
   │        └── val2017
   │        └── ...
   >Frido
   └── configs
   │   └── t2i
   │   └── ... 
   └── exp
   │   └── t2i
   │        └── frido_f16f8
   │             └── checkpoints
   │                  └── model.ckpt
   │   └── layout2i
   │   └── ...
   └── frido
   └── scripts
   └── tools
   └── ...
   ```

   </details>

---

## Download pre-trained models
The following table describs tasks and models that are currently available. 
To auto-download all model checkpoints of Frido, please use following command:
```bash
bash tools/download.sh
```

| Task                   | Datase                     | FID    | Comments
| ---------------------- | -------------------------- | ------ | -------------
| Text-to-image          | COCO 2014                  | 11.24  | 
| Text-to-image (mini)   | COCO 2014                  | 64.85  | 1000 images of mini-val; FID was calculated against corresponding GT images.
| Layout-to-image        | COCO (finetuned OpenImage) | 37.14  | FID calculated on 2,048 val images.
| Layout-to-image (mini) | COCO (finetuned OpenImage) | 122.48 | 500 images of mini-val; FID was calculated against corresponding GT images.

> *The mini-versions are for quick testing and reproducing, which can be done within 1 hours on 1*V100. High FID is expected. To evaluate generation quality, full validation / test split needs to be run.*

> *FID scores were evaluated by using [torch-fidelity](https://github.com/toshas/torch-fidelity). The scores may slightly fluctuate due to the inherent initial random noise of diffusion models.*

---

## Inference Frido
We now provide scripts for testing Frido. (Full training code will be released soon.)

### Quick Start
Please checkout the jupyter notebook [`demo.ipynb`](https://github.com/davidhalladay/Frido/blob/main/demo.ipynb) for a simple demo on text-to-image generation for COCO.


Once the datasets and model weights are properly set up, one may test Frido by the following commands.
### Text-to-image 
```bash
# for full validation:
bash tools/eval_t2i.sh

# for mini-val:
bash tools/eval_t2i_minival.sh

```
 - Default output folder will be `exp/t2i/frido_f16f8/samples`
### Layout-to-image

```bash
# for full validation:
bash tools/eval_layout2i.sh

# for mini-val:
bash tools/eval_layout2i_minival.sh
```
Default output folder will be `exp/layout2i/frido_f8f4/samples`

(Optional) You can modify the script by adding following augments. 

- -o [OUTPUT_PATH]  :  to change the output folder path.

- -c [INT]  :  number of steps for ddim and fastdpm sampling. (default=200)

### Multi-GPU testing

We provide code for multiple GPUs testing. Please refer to scripts of [`tools/eval_t2i_multiGPU.sh`](https://github.com/davidhalladay/Frido/blob/main/tools/eval_t2i_multiGPU.sh)

For example, 4-gpu inference can be run by the following.

```bash
bash eval_t2i_multiGPU.sh 4
```

---

## Evaluation
FID scores were evaluated by using [torch-fidelity](https://github.com/toshas/torch-fidelity).

After running inference, FID score can be computed by the following command:
```bash
fidelity --gpu 0 --fid --input2 [GT_FOLDER] --input1 [PRED_FOLDER]
```
Example:
```bash
fidelity --gpu 0 --fid --input2 exp/t2i/frido_f16f8/samples/.../img/inputs --input1 exp/t2i/frido_f16f8/samples/.../img/sample
```

## Acknowledgement
We build Frido codebase heavily on the codebase of [Latent Diffusion Model (LDM)](https://github.com/CompVis/latent-diffusion) and [VQGAN](https://github.com/CompVis/taming-transformers). We sincerely thank the authors for open-sourcing! 

## Citation
If you find this code useful for your research, please consider citing:
```bibtex
@article{fan2022frido,
  title={Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis},
  author={Fan, Wan-Cyuan and Chen, Yen-Chun and Chen, Dongdong and Cheng, Yu and Yuan, Lu and Wang, Yu-Chiang Frank},
  journal={arXiv preprint arXiv:2208.13753},
  year={2022}
}
```

## License

MIT
