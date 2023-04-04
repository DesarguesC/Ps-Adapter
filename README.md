# Ps-Adapter

Now we've developped a new method to train pretrained adapters jointly to obtain an extra adapter which will be much stabler.

The project is based on Tencent T2I-Adapter.

## Packages

install packages as required by
```bat
pip install -r requirements.txt
```
Install nlpconnct/vit-gpt2-image-captioning
```bat
git lfs install
git clone https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
```

## Datasets
If you'd like to train the Ps-Adapter on your own dataset, there's only a need of images without lable.

All of images format '.png', '.jpg', '.jpeg', '.webp' are allowed to be put in route Datasets/sets.


you can download only MPII-Human Pose dataset by running
```bat
kaggle datasets download -d harshpatel66/mpii-human-pose
```



p.s.: You can download your image dataset (e.g. zip, tar, gz) onto Datasets/sets and hen unzip then direcly. By running
```bat
cd Datasets
python deal.py
```
all images under folder Datasets/sets will we moved into Datasets/Data

After that, run
```bat
cd ..
python caption_pose.py --max_length max_length --num_beams num_beams
```
to get captions and kepose images. Here, "max_length" and "num_beams" are parameters from [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) program. So, make sure vit-gpt2 model has been installed.


After these, cpaions will be written into "captions.csv" under folder "./Datasets/Captions/" and images will be gathered under folder "./Datasets/Data". 

## Train

Before training, set environment variables as follow
```bat
export RANK=0
export WORLD_SIZE=2     # the number of the thread to be called
export MASTER_ADDR=localhost
export MASTER_PORT=5678
```

try to train
```bat
python train_ps_adapter.py --local_rank=0 --gpus 0 --num_workers 2      # with single gpu
```



