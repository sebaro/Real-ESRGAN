
## General Image/Video Restoration

▶️ [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/)

## Face Restoration

▶️ [GFPGAN](https://github.com/TencentARC/GFPGAN)<br>
▶️ [CodeFormer](https://github.com/sczhou/CodeFormer)<br>
▶️ [RestoreFormer](https://github.com/wzhouxiff/RestoreFormerPlusPlus)

## Dependencies

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

## Installation

1. Clone repo

    ```bash
    git clone https://github.com/sebaro/Real-ESRGAN.git
    cd Real-ESRGAN
    ```

1. Install dependent packages

    ```bash
    pip install -r requirements.txt
    ```

## Inference

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input image or folder
  -o OUTPUT, --output OUTPUT
                        Output folder
  -n MODEL_NAME, --model_name MODEL_NAME
                        Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3 |
                        realesr-general-x4v3. Default: RealESRGAN_x4plus
  -dn DENOISE_STRENGTH, --denoise_strength DENOISE_STRENGTH
                        Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. Only used for the realesr-general-x4v3 model
  -s OUTSCALE, --outscale OUTSCALE
                        The final upsampling scale of the image
  --model_path MODEL_PATH
                        [Option] Model path. Usually, you do not need to specify it
  --suffix SUFFIX       Suffix of the restored image
  -t TILE, --tile TILE  Tile size, 0 for no tile during testing
  --tile_pad TILE_PAD   Tile padding
  --pre_pad PRE_PAD     Pre padding size at each border
  --pre_enhance         Enhance image before upscaling
  --face_enhance        Use GFPGAN to enhance face
  --face_enhance_arch FACE_ENHANCE_ARCH
                        Face enhance arch: CodeFormer | GFPGAN | RestoreFormer | RestoreFormer++. Default: CodeFormer
  --face_enhance_model FACE_ENHANCE_MODEL
                        Face enhance model for GFPGAN: 1.4 | 1.3 | 1.2. Default: 1.4
  --face_enhance_fidelity FACE_ENHANCE_FIDELITY
                        Face enhance fidelity. Default: 1
  --face_upsample       Face upsample after enhancement. Default: False
  --face_upsample_model FACE_UPSAMPLE_MODEL
                        Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3 |
                        realesr-general-x4v3. Default: RealESRGAN_x4plus
  --face_offset FACE_OFFSET
                        Face offset for more precise back alignment. Default: 0
  --fp32                Use fp32 precision during inference. Default: fp16 (half precision).
  --alpha_upsampler ALPHA_UPSAMPLER
                        The upsampler for the alpha channels. Options: realesrgan | bicubic
  --ext EXT             Image extension. Options: auto | jpg | png, auto means using the same extension as inputs
  -g GPU_ID, --gpu-id GPU_ID
                        gpu device to use (default=None) can be 0,1,2 for multi-gpu
```
