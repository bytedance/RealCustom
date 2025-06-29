#!/bin/bash

$HOME/anaconda3/bin/conda create --name realcustom python=3.10 -y
source $HOME/anaconda3/bin/activate realcustom
echo "The virtual environment 'realcustom' has been created and activated with Python version 3.10."

echo "Install opencv dependencies"
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y

echo "Install pip dependencies"
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r envs/requirements.txt

# install clip
pip install git+https://github.com/openai/CLIP.git
pip install open_clip_torch