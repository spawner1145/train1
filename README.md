# Naifu

naifu (or naifu-diffusion) is designed for training generative models with various configurations and features. The code in the main branch of this repository is under development and subject to change as new features are added.

## Installation

To get started with Naifu, follow these steps to install the necessary dependencies:

```bash
# Clone the Naifu repository:
git clone --depth 1 https://github.com/mikubill/naifu

# Install the required Python packages:
cd naifu && pip install -r requirements.txt
```

Make sure you have a compatible version of Python installed (Python 3.9 or above).

## Usage

Naifu provides a flexible and intuitive way to train models using various configurations. To train a model, use the trainer.py script and provide the desired configuration file as an argument.

```bash
python trainer.py --config config/<config_file>

# or (same as --config)
python trainer.py config/<config_file>
```

Replace `<config_file>` with one of the available configuration files listed below.

## Configurations

Choose the appropriate configuration file based on training objectives and environment.

Train SDXL (Stable Diffusion XL) model
```bash
# prepare image data (to latents)
python scripts/encode_latents_xl.py -i <input_path> -o <encoded_path>

# sd_xl_base_1.0_0.9vae.safetensors
python trainer.py config/train_sdxl.yaml

# For huggingface model support
# stabilityai/stable-diffusion-xl-base-1.0
python trainer.py config/train_diffusers.yaml

# use original sgm loss module
python trainer.py config/train_sdxl_original.yaml
```

Train SDXL refiner (Stable Diffusion XL refiner) model
```bash
# stabilityai/stable-diffusion-xl-refiner-1.0
python trainer.py config/train_refiner.yaml
```

Train original Stable Diffusion 1.4 or 1.5 model
```bash
# runwayml/stable-diffusion-v1-5
# Note: will save in diffusers format
python trainer.py config/train_sd15.yaml
```

Train SDXL model with LyCORIS.
```bash
# Based on the work available at KohakuBlueleaf/LyCORIS
pip install lycoris_lora toml
python trainer.py config/train_lycoris.yaml
```

Use fairscale strategy for distributed data parallel sharded training
```bash
pip install fairscale
python trainer.py config/train_fairscale.yaml
```

Train SDXL model with Diffusion DPO  
Paper: Diffusion Model Alignment Using Direct Preference Optimization ([arxiv:2311.12908](https://arxiv.org/abs/2311.12908))
```bash
# dataset: yuvalkirstain/pickapic_v2
# Be careful tuning the resolution and dpo_betas!
# will save in diffusers format
python trainer.py config/train_dpo_diffusers.yaml # diffusers backend
python trainer.py config/train_dpo.yaml # sgm backend
```

Train Pixart-Alpha model  
Paper: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis ([arxiv:2310.00426](https://arxiv.org/abs/2310.00426))
```bash
# PixArt-alpha/PixArt-XL-2-1024-MS
python trainer.py config/train_pixart.yaml
```

Train SDXL-LCM model  
Paper: Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference ([arxiv:2310.04378](https://arxiv.org/abs/2310.04378))
```bash
python trainer.py config/train_lcm.yaml
```

Train StableCascade model ([Sai](https://github.com/Stability-AI/StableCascade/))
```bash
# currently only stage_c (w/ or w/o text encoder)
python trainer.py config/train_cascade_stage_c.yaml
```

Train GPT2 model
```bash
# currently only stage_c (w/ or w/o text encoder)
python trainer.py config/train_gpt2.yaml
```

Train with [Phi-1.5/2](https://huggingface.co/microsoft) model
```bash
python trainer.py config/train_phi2.yaml
```

Train language models ([LLaMA](https://github.com/facebookresearch/llama), [Qwen](https://huggingface.co/Qwen), [Gemma](https://huggingface.co/google) etc)
```bash
# Note that prepare data in sharegpt/chatml format, or define your own dataset in data/text_dataset.py
# See example dataset for reference: function-calling-sharegpt
python trainer.py config/train_general_llm.yaml
```

Train language models with lora or qlora (For example, [Mistral](https://huggingface.co/mistralai))
```bash
python trainer.py config/train_mistral_lora.yaml
```

## Other branches

* sgm - Uses the [sgm](https://github.com/Stability-AI/generative-models) to train SDXL models.
* sd3 - Trainer for SD3 models - use with caution: may produce undesired result
* hydit - Trainer for hunyuan dit models (v1.1 and v1.2)
* main-archived - Contains the original naifu-diffusion code for training Stable Diffusion 1.x models.

For branches without documentation, please follow the installation instructions provided above.

## Lumina2 训练蒙版损失（Loss Mask）

你可以为 Lumina2 训练提供一张与原图一一对应的“蒙版图片”，用于按像素加权训练损失：

- 白色（1.0）表示该区域计算损失；
- 黑色（0.0）表示不计算损失；
- 灰度值表示不同的损失权重（0.0~1.0）。

启用方法：

1) 在训练配置里为数据集增加以下可选项（示例见 `config/train_lumina2_test.yaml`）：

```
dataset:
	# ... 其它数据集参数
	use_loss_mask: true
	loss_mask_key: "mask"  # Arrow/索引文件中对应蒙版图片的列名
```

2) 数据加载与图像的裁剪/缩放/随机水平翻转完全一致，蒙版与图像严格对齐；

3) 蒙版图片可以是单通道（推荐）或 RGB（会自动转换为灰度），取值范围将被归一化到 [0,1]；

4) 训练时蒙版会被按 VAE 潜空间分辨率下采样并用于对像素级 MSE 进行加权平均；未提供蒙版时，行为与原始训练一致。

### 数据准备与列名约定

- 数据索引（Arrow/IndexKits）中需要存在一列用于存放蒙版图片的字段，例如 `mask`；
- 该列的图片与图像列一一对应（同一条样本下），尺寸可以不同，加载时会自动按与原图相同的策略 resize/crop/flip；
- 推荐使用无损或较少损失的格式（PNG、WebP-lossless），避免 JPEG 压缩导致灰度值偏移；
- 蒙版可为：
	- 单通道灰度图（首选）；
	- RGB 彩色图（会自动按 0.299R+0.587G+0.114B 转灰度）；
	- 取值范围会被规范到 [0,1]；白=1，黑=0，灰=中间权重。

### 行为细节（对齐与加权）

- 几何变换对齐：
	- multireso=true：使用桶的 `resize_and_crop`，图与蒙版共享同一个裁剪位置与尺寸；
	- multireso=false：按等比例 resize 后随机裁剪，图与蒙版共享同一随机裁剪位置；
	- 随机水平翻转：图与蒙版共享同一个随机 flip 决策；
- 通道与尺度：
	- 蒙版在图像空间为 [B,1,H,W]；
	- 在前向中按 VAE 潜空间大小（例如 H/8,W/8）下采样至 [B,1,h,w]；
	- 在像素级 MSE 中对所有潜向量通道统一加权（等效于对 C 维取平均后再按 [h,w] 权重做加权均值）。

### 边界情况与性能

- 缺失/读取失败的蒙版：降级为全 1 蒙版（等价于不加权），并在日志中提示；
- 全黑蒙版：分母会加上极小 eps 以避免除零，等价于极小权重；
- 性能影响：蒙版仅单通道，额外显存/计算开销很小；
- 兼容性：不影响采样器/分布式/梯度累计等流程。

### 常见问题（FAQ）

- Q: 我只想对人脸/主体区域强化训练，怎么做？
	- A: 先用分割/检测模型生成对应区域灰度蒙版，值域在 0~1（建议软权重），写入 `loss_mask_key` 对应列即可。
- Q: 蒙版与图对不齐？
	- A: 本实现严格共享相同的 resize/crop/flip 参数；若仍错位，请核对数据是否一一对应，以及是否走了同一套增强链路。
- Q: 想做二值化？
	- A: 目前默认不做阈值二值化；你可在数据写入前自行二值化，或根据需要后续增加一个可选阈值参数。
