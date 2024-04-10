## WaDiGAN-SR: A Wavelet-based Diffusion GAN approach to Image Super-Resolution ##

<div align="center">
    <a href="https://colab.research.google.com/drive/1EHcwoRwpzC5NTJVOe6rTPv6fN1yxSQ5X?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a>
    <br>
    <a href="#installation"> Installation </a> | <a href="#dataset-preparation"> Dataset preparation </a> | <a href="#how-to-run">How to run</a> 
    <br>
    <a href="#results">Results</a> | <a href="#evaluation">Evaluation</a> | <a href="#acknowledgments">Acknowledgments</a> 
    <br>
    <br>
</div>

### Abstract ###
In recent years, Diffusion Models have emerged as a superior alternative to Generative Adversarial Networks (GANs) for high-fidelity image generation, with wide applications in text-to-image generation, image-to-image translation, and super-resolution. However, their real-time feasibility is hindered by slow training and inference speeds. This study addresses this challenge by proposing a wavelet-based conditional Diffusion GAN scheme for Single-Image Super-Resolution (SISR). Our approach utilizes the Diffusion GAN paradigm to reduce the number of timesteps required by the reverse diffusion process and the Discrete Wavelet Transform (DWT) to achieve dimensionality reduction, decreasing training and inference times significantly. The results of an experimental validation on the CelebA-HQ dataset confirm the effectiveness of our proposed scheme. Our approach outperforms other state-of-the-art methodologies successfully ensuring high-fidelity output while overcoming inherent drawbacks associated with diffusion models in time-sensitive applications.

<p align="left">
  <img src="./assets/backward_diff_proc.png" width="700" alt="Alt Text">
</p>


## Installation ##
Python `3.10.12` and Pytorch `1.11.0` are used in this implementation.

You can install neccessary libraries as follows:
```bash
pip install -r requirements.txt
```

## Dataset preparation ##
We trained on CelebA HQ (16x16 -> 128x128). 

If you don't have the data, you can prepare it in the following way:

Download [CelebaHQ 256x256](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256).

Use the following script to prepare the dataset in PNG or LMDB format:  
**IMPORTANT**: be sure to have the images in the folder sequentially numbered, otherwise the conversion into LMDB won't work.
```
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
# Specify -l for LMDB format
python datasets_prep/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```

Once a dataset is downloaded and prepared, please put it in `data/` directory as follows:
```
data/
├── celebahq_16_128
```

## How to run ##
We provide a bash script for our experiments. The syntax is following:
```
bash run.sh <DATASET> <MODE> <#GPUS>
```
where: 
- `<DATASET>`: `celebahq_16_128`.
- `<MODE>`: `train` and `test`.
- `<#GPUS>`: the number of gpus (e.g. 1, 2, 4, 8).

Note, please set argument `--exp` correspondingly for both `train` and `test` mode. All of detailed configurations are well set in [run.sh](./run.sh). 

**GPU allocation**: Our work is experimented on a single NVIDIA Tesla T4 GPU 15GBs.


## Results ##
Comparisons between our model, SR3, DiWa and ESRGAN (all of them trained on 25k iteration steps) are below:
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>ESRGAN</th>
      <th>SR3</th>
      <th>DiWa</th>
      <th><strong>Ours</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PSNR ↑</td>
      <td><u>21.13</u></td>
      <td>14.65</td>
      <td>13.68</td>
      <td><strong>23.38</strong></td>
    </tr>
    <tr>
      <td>SSIM ↑</td>
      <td><u>0.59</u></td>
      <td>0.42</td>
      <td>0.13</td>
      <td><strong>0.68</strong></td>
    </tr>
    <tr>
      <td>LPIPS ↓</td>
      <td><u>0.082</u></td>
      <td>0.365</td>
      <td>0.336</td>
      <td><strong>0.061</strong></td>
    </tr>
    <tr>
      <td>FID ↓</td>
      <td><strong>20.8</strong></td>
      <td>99.4</td>
      <td>270</td>
      <td><u>47.2</u></td>
    </tr>
  </tbody>
</table>

The checkpoint we used to compute these results is provided [here]().

Inference time is computed over 300 trials on a single NVIDIA Tesla T4 GPU for a batch size of 64.

Downloaded pre-trained models should be put in `saved_info/srwavediff/<DATASET>/<EXP>` directory where `<DATASET>` is defined in [How to run](#how-to-run) section and `<EXP>` corresponds to the folder name of pre-trained checkpoints.

<table>
  <thead>
    <tr>
      <th></th>
      <th>ESRGAN</th>
      <th>SR3</th>
      <th>DiWa</th>
      <th><strong>Ours</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Runtime</td>
      <td>0.04s</td>
      <td>60.3s</td>
      <td>34.7s</td>
      <td>0.12s</td>
    </tr>
    <tr>
      <td>Parameters</td>
      <td>31M</td>
      <td>98M</td>
      <td>92M</td>
      <td>57M</td>
    </tr>
  </tbody>
</table>


## Evaluation ##
FID, PSNR, SSIM and LPIPS are computed on the whole test-set (6000 samples).

### Inference ###
Samples can be generated by calling [run.sh](./run.sh) with `test` mode.

### FID ###
To compute FID of pretrained models at a specific epoch, we can add additional arguments including ```--compute_fid``` and ```--real_img_dir /path/to/real/images``` of the corresponding experiments in [run.sh](./run.sh).

### PSNR, SSIM and LPIPS ###
A simple script is provided to compute PSNR, SSIM and LPIPS for the results. Please notice that you have to run inference without the ```--compute_fid``` and ```--measure_time``` options before executing the script.
```
python /benchmark/eval.py -p [result root]
```

## Acknowledgments
- WaveDiff [code](https://github.com/VinAIResearch/WaveDiff) and [paper](https://arxiv.org/abs/2211.16152);
- SR3 [unofficial code](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/master) and [paper](https://arxiv.org/abs/2104.07636);
- Wavelet transforms [code1](https://github.com/LiQiufu/WaveCNet) and [code2](https://github.com/fbcotter/pytorch_wavelets);
- DiWa [paper](https://arxiv.org/abs/2304.01994);
- ESRGAN [paper](https://arxiv.org/abs/1809.00219);
- BasicSR [code](https://github.com/XPixelGroup/BasicSR) for its ESRGAN implementation.



