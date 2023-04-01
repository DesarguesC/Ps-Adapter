#!bin/sh
chmod 600 kaggle
kaggle -v
pip show kaggle
kaggle datasets download -d harshpatel66/mpii-human-pose
kaggle datasets download -d aneesh10/cricket-shot-dataset
kaggle datasets download -d tr1gg3rtrash/yoga-posture-dataset

unzip mpii-human-pose.zip
unzip cricket-shot-dataset.zip
unzip yoga-posture-dataset.zip

cd ..
conda activate ldm
python deal.py
cd ..
python deal.py

