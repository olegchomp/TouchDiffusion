# TouchDiffusion
<a href="https://discord.com/invite/wNW8xkEjrf"><img src="https://discord.com/api/guilds/838923088997122100/widget.png?style=shield" alt="Discord Shield"/></a>

TouchDesigner implementation for real-time Stable Diffusion interactive generation with [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion).

**Benchmarks with stabilityai/sd-turbo, 512x512 and 1 batch size.**

| GPU | FPS |
| --- | --- |
| 4090 | 55-60 FPS |
| 4080 | 47 FPS |
| 3090ti | 37 FPS |
| 3090 | 30-32 FPS |
| 4070 Laptop | 24 FPS |
| 3060 12GB | 16 FPS |

## Disclaimer
**Notice:** This repository is in an early testing phase and may undergo significant changes. Use it at your own risk. 

## Usage
> [!TIP]
> TouchDiffusion can be installed in multiple ways. **Portable version** have prebuild dendencies, so it prefered way to install or **Manuall install** is step by step instruction.

#### Portable version:
Includes preinstalled configurations, ensuring everything is readily available for immediate use.
1. Download and extract [archive](https://boosty.to/vjschool/posts/39931cd6-b9c5-4c27-93ff-d7a09b0918c5?share=post_link)
2. Run ```webui.bat```. It will provide url to web interface (ex. ```http://127.0.0.1:7860```)
3. Open ```install & update``` tab and run ```Update dependencies```.
   
#### Manuall install:
You can follow [YouTube tutorial](https://youtu.be/3WqUrWfCX1A)

Required TouchDesigner 2023 & Python 3.11
1. Install [Python 3.11](https://www.python.org/downloads/release/python-3118/)
2. Install [Git](https://git-scm.com/downloads)
3. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) 11.8 (required PC restart)
4. Download [TouchDiffusion](https://github.com/olegchomp/TouchDiffusion/archive/refs/heads/main.zip).
5. Open ```webui.bat``` with text editor and set path to Python 3.11 in ```set PYTHON_PATH=```. (ex. ```set PYTHON_PATH="C:\Program Files\Python311\python.exe"```)
6. Run ```webui.bat```. After installation it will provide url to web interface (ex. ```http://127.0.0.1:7860```)
7. Open ```install & update``` tab and run ```Update dependencies```. (could take ~10 minutes, depending on your internet connection)
8. If you get pop up window with error related to .dll, run ```Fix pop up```
9. Restart webui.bat

#### Accelerate model:
Models in ```.safetensors``` format must be in ```models\checkpoints``` folder. (as for sd_turbo, it  will be auto-downloaded).

**Internet connection required, while making engines.**

1) Run ```webui.bat```
2) Select model type. 
3) Select model.
4) Set width, height and amount of sampling steps (Batch size)
5) Select acceleration lora if available.
6) Run ```Make engine``` and wait for acceleration to finish. (could take ~10 minutes, depending on your hardware)

#### TouchDesigner inference:
1. Add **TouchDiffusion.tox** to project
2. On ```Settings``` page change path to ```TouchDiffusion``` folder (same as where webui.bat).
3. Save and restart TouchDesigner project.
4. On ```Settings``` page select Engine and click **Load Engine**.
5. Connect animated TOP to input. Component cook only if input updates. 

#### Known issues / Roadmap:
- [ ] Fix Re-init. Sometimes required to restart TouchDesigner for initializing site-packages.
- [ ] Code clean-up and rework.
- [x] Custom resolution (for now fixed 512x512)
- [ ] CFG not affecting image
- [ ] Add Lora
- [ ] Add Hyper Lora support
- [ ] Add ControlNet support
- [ ] Add SDXL support

## Acknowledgement
Based on the following projects:
* [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) - Pipeline-Level Solution for Real-Time Interactive Generation
* [TopArray](https://github.com/IntentDev/TopArray) - Interaction between Python/PyTorch tensor operations and TouchDesigner TOPs.
