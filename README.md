# Ps-Adapter

Now we've developped a new method to train pretrained adapters jointly to obtain an extra adapter which will be much stabler.

The project is based on Tencent T2I-Adapter.

## Packages

create environment
```bat
conda env create -f environment.yaml
```

install packages as required by
```bat
pip install -r requirements.txt
```
Install nlpconnct/vit-gpt2-image-captioning
```bat
git lfs install
git clone https://huggingface.co/nlpconnct/vit-gpt2-image-captioning
```
And remember to put T2I-Adapter as well as stable-diffusion ckpt in folder 'models'.

fetch whole clip model in folder 'models'

```bat
git clone https://huggingface.co/openai/clip-vit-large-patch14
```


## Datasets
If you'd like to train the Ps-Adapter on your own dataset, there's only a need of images without lable.

All of images format '.png', '.jpg', '.jpeg', '.webp' are allowed to be put in route Datasets/sets.


you can download only MPII-Human Pose dataset by running
```bat
cd Datasets
kaggle datasets download -d harshpatel66/mpii-human-pose
```
download Cricket Shot Dataset
```bat
kaggle datasets download -d aneesh10/cricket-shot-dataset
```




p.s.: You can download your image dataset (e.g. zip, tar, gz) onto Datasets/sets and hen unzip then direcly. By running
```bat
cd Datasets
python deal.py --source <your dataset folder> --target "YourRoot/Ps-Adapter/Datasets/Data"
```
all images under folder Datasets/sets will we moved into Datasets/Data

After that, run
```bat
cd ..
python deal.py --length max_length --beams num_beams --random_num
```
or
```bat
python recaption.py --length 77 --beams 10 --image ~/autodl-tmp/self/Datasets/mpii/images --outdir_captions ~/autodl-tmp/Data/resaption/Captions --outdir_keypose ~/autodl-tmp/Data/resaption/Keypose
```
(no '/' after the ending folder)


to get captions and kepose images. Here, "max_length" and "num_beams" are parameters from [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) program. So, make sure vit-gpt2 model has been installed.


After these, cpaions will be written into "captions.csv" under folder "./Datasets/Captions/" and images will be gathered under folder "./Datasets/Data". 

## Train

Before training, set environment variables as follow
```bat
export RANK=0
export WORLD_SIZE=2     # the number of the thread to be called
export MASTER_ADDR=localhost
export MASTER_PORT=5678
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=...(your gpus rank)
```


Use single gpu to train
```bat
python single_train.py --sd_ckpt ~/autodl-tmp/models/v1-5-pruned.ckpt --adapter_ori ~/autodl-tmp/models/t2iadapter_openpose_sd14v1.pth --adapter_ckpt ~/autodl-tmp/models/t2iadapter_openpose_sd14v1.pth  --caption_path ~/autodl-tmp/Datasets/Captions/captions.csv --keypose_folder ~/autodl-tmp/Datasets/Keypose/ --resize yes --bsize 4
```


However, if you want to use multi gpus for training, you should first ensure the nccl environment

Build:
```bat
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build  # ensure your CUDA was installed on /usr/local/cuda, otherwise, run: make -j src.build CUDA_HOME=<YOUR CUDA PATH>
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"   # choosable; it might take several minutes
```
Install on Ubuntu or Debian
```bat
sudo apt install build-essential devscripts debhelper
make pkg.debian.build
ls build/pkg/deb/     # test
```

After this, test the installation
```bat
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <YOUR GPU AMOUNT>
```

Use multi gpus to train
```bat
python -m torch.distributed.launch --nproc_per_node=<gpu ammounts> --master_port=5678  train_ps_adapter.py --sd_ckpt ~/autodl-tmp/models/v1-5-pruned.ckpt --adapter_ori ~/autodl-tmp/models/t2iadapter_openpose_sd14v1.pth --adapter_ckpt ~/autodl-tmp/models/t2iadapter_openpose_sd14v1.pth  --caption_path ~/autodl-tmp/Datasets/Captions/captions.csv --keypose_folder ~/autodl-tmp/Datasets/Keypose/ --resize yes --bsize 4 --local_rank 0 --gpus <your gpu amounts> --num_workers <max: your cpu kernel> * 2
```





