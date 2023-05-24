# pretreatment of the (input, output) data

import json
import cv2
import pandas as pd
import numpy as np
import os
from basicsr.utils import img2tensor
from random import randint, shuffle
from ldm.util import resize_numpy_image as rs
from einops import rearrange

Inter = {
    'inter_cubic': cv2.INTER_CUBIC,
    'inter_nearest': cv2.INTER_NEAREST,
    'inter_linear': cv2.INTER_LINEAR,
    'inter_lanczos4': cv2.INTER_LANCZOS4
}

def deal(Input):
    sh = Input.shape
    assert len(sh) == 3
    arr = []
    for i in range(3):
        if sh[i]==3:
            for j in range(3):
                if j!=i:
                    arr.append(j)
            arr.append(i)
            break
    
    assert len(arr) == 3
    assert sh[arr[2]] == 3, f'problem occurred with Input shape: {Input.shape}'
    Input = Input.reshape(sh[arr[0]], sh[arr[1]], sh[arr[2]])
    return Input


class PsKeyposeDataset():
    def __init__(self, caption_path, keypose_path, resize=False, interpolation="inter_cubic", factor=1, max_resolution=512*512):
        # caption_path: csv file path -> read csv file
        # keypose_path: image folder -> store image names
        self.caption_path = caption_path
        self.keypose_path = keypose_path if keypose_path.endswith('/') else keypose_path + '/'
        self.resize = resize
        self.inter = interpolation
        self.factor = factor
        self.item_shape = (512, 512)
        self.max_resolution = max_resolution
        

        super(PsKeyposeDataset, self).__init__()
        try:
            prompt_list = list(pd.read_csv(caption_path)['CAPTIONS'])
        except:
            raise Exception('Read caption failed. Possible reason: "CAPTIONS" does not exists')

        image_list= os.listdir(keypose_path)
        length = len(image_list)
        assert length == len(prompt_list), 'Data preprocessing wrong. image_num: {0}, prompt_num: {1}'.format(length, len(prompt_list))

        # image: OpenKeypose image

        self.files = []
        print('Binding Training Dataset...')
        for A in range(length):
            for B in range(0, A):
                self.files.append({
                     'primary': image_list[A],
                     'secondary': image_list[B],
                     'prompt': prompt_list[A]
                     })
                self.files.append({
                     'primary': image_list[B],
                     'secondary': image_list[A],
                     'prompt': prompt_list[B]
                    })
        shuffle(self.files)
        
        A = rs(cv2.imread(self.keypose_path+self.files[randint(1,100)]['primary']), max_resolution=self.max_resolution, resize_method=Inter[self.inter])
        print('base data shape = ', A.shape)
        h, w, _ = A.shape
        
        self.item_shape = h, w
        
        # self.item_shape = (64, 96)
        
        
        
        

    def __getitem__(self, idx):
        file = self.files[idx]
        assert isinstance(file, dict)
        read_img = lambda x: np.array(x / 255., dtype=np.float32) # img2tensor(x, bgr2rgb=True, float32=True) / 255.
        
        A, B = cv2.imread(self.keypose_path+file['primary']), cv2.imread(self.keypose_path+file['secondary'])
        # read
        A = deal(A)
        B = deal(B)
        # regular
        
        h, w = self.item_shape
  
        B = cv2.resize(B, (w, h), interpolation=Inter[self.inter])
        A = cv2.resize(A, (w, h), interpolation=Inter[self.inter])
        # (h, w) or (w, h) ?
        
        assert A.shape == B.shape, 'two keypose must have same shape: Shape1-{0}, Shape2-{1}'.format(A.shape, B.shape)

        prompt = file['prompt'].strip()
        A = read_img(A)
        B = read_img(B)    
        A = rearrange(A, 'u v w -> w u v')
        B = rearrange(B, 'u v w -> w u v')
        
        return {
            'primary': A,
            'secondary': B,
            'prompt': prompt
        }

    def __len__(self):
        return len(self.files)