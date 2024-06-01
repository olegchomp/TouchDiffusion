# TouchDiffusion
<a href="https://discord.com/invite/wNW8xkEjrf"><img src="https://discord.com/api/guilds/838923088997122100/widget.png?style=shield" alt="Discord Shield"/></a>

TouchDesigner implementation (up to 2X FPS increase) for real-time Stable Diffusion interactive generation with [StreamDiffusion](https://pages.github.com/](https://github.com/cumulo-autumn/StreamDiffusion)https://github.com/cumulo-autumn/StreamDiffusion).

## Disclaimer
**Notice:** This repository is in an early testing phase and may undergo significant changes. Use it at your own risk. 

**Use #touchdiffusion forum in [discord](https://discord.com/invite/wNW8xkEjrf) for submiting issues.**

## Usage
Required TouchDesigner 2023 & Python 3.11

#### Install:
1. Install [Python 3.11](https://www.python.org/downloads/release/python-3118/)
2. Install [Git](https://git-scm.com/downloads)
3. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) 11.8
4. Download [latest release](https://github.com/olegchomp/TouchDiffusion/releases)
5. Run ```webui.bat```

On first run it will create .venv and install dependencies 

#### Accelerate model:
Models must be in ```models``` folder or you can set [HF_HOME](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) in system variables to change this folder.

1) Run ```webui.bat```
2) Select model or provide model name (will be downloaded if not exists)
4) Set amount of sampling steps (Batch size)
5) Select Turbo (if model is Turbo), LCM (Add LCM Lora), None (for other types)
6) Click submit and wait for acceleration to finish

#### TouchDesigner inference:
1. Add **TouchDiffusion.tox** to project
2. On ```Settings``` page change path to ```TouchDiffusion``` folder (same as where webui.bat) and click **Re-init**.
3. On ```Settings``` page select Engine, Acceleration, Batch size (values locked to avaliable for engines) and click **Load Engine**.

#### Known issues / Roadmap:
* Fix Re-init. Sometimes required to restart TouchDesigner for initializing site-packages.
* Code clean-up and rework.
* Custom resolution (for now fixed 512x512)
* CFG not affecting image
* Add Lora
* Add Hyper Lora support
* Add ControlNet support
* Add SDXL support

## Acknowledgement
Based on the following projects:
* [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) - Pipeline-Level Solution for Real-Time Interactive Generation
* [TopArray](https://github.com/IntentDev/TopArray) - Interaction between Python/PyTorch tensor operations and TouchDesigner TOPs.
