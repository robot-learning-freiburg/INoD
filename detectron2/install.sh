#!/usr/bin/env bash
# install dependencies
pip3 install pandas==2.0.3
pip3 install pascal-voc-writer==0.1.4
pip3 install imagehash==4.3.1
pip3 install wandb==0.16.0
pip3 install opencv-python==4.8.1.78
pip3 install albumentations==1.3.1
pip3 install Shapely==2.0.2
# use torch 1.11. or below, detectron2 has issues with torch 1.12.
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torchmetrics==1.2.0
pip3 install setuptools==59.5.0
pip3 install Pillow==9.5.0
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo "Installations finished."
