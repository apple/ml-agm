#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r setup/requirement.txt 
python download_datasets.py
pip install -U pytorch_warmup
pip install --upgrade wandb
# # AFHQ download
URL=https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
ZIP_FILE=./dataset/afhq_v2.zip
mkdir -p ./dataset/afhqv2
wget --wait 10 --random-wait --continue  -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./dataset/afhqv2
rm $ZIP_FILE
# AFHQ download
python edm/dataset_tool.py --source=dataset/afhqv2 --dest=dataset/afhqv2-64x64.zip --resolution=64x64