# pretreatment of the (input, output) data

import json
import cv2
import pandas as pd
import os
from basicsr.utils import img2tensor
from random import randint, shuffle
from ldm.util import resize_numpy_image
from einops import rearrange

Inter = {
    'inter_cubic': cv2.INTER_CUBIC,
    'inter_nearest': cv2.INTER_NEAREST,
    'inter_linear': cv2.INTER_LINEAR,
    'inter_lanczos': cv2.INTER_LANCZOS4
}

def deal(Input):
    t = Input.shape[0]
    s = t if not t==3 else Input.shape[1]
    Input = Input.reshape(t, -1, 3)
    return Input

# def divide(shape, factor):
#     return (shape[0] // factor, shape[1] // factor)


class PsKeyposeDataset():
    def __init__(self, caption_path, keypose_path, resize=False, interpolation="inter_cubic", factor=1):
        # caption_path: csv file path -> read csv file
        # keypose_path: image folder -> store image names
        self.caption_path = caption_path
        self.keypose_path = keypose_path if keypose_path.endswith('/') else keypose_path + '/'
        self.resize = resize
        self.inter = interpolation
        self.factor = factor
        self.flag = 0

        super(PsKeyposeDataset, self).__init__()
        try:
            prompt_list = list(pd.read_csv(caption_path)['CAPTIONS'])
        except:
            raise Exception('Read caption failed. Possible reason: "CAPTIONS" does not exists')

        image_list= os.listdir(keypose_path)
        length = len(image_list)
        assert length == len(prompt_list), 'Data preprocessing wrong.'

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

    def __getitem__(self, idx):
        file = self.files[idx]
        assert isinstance(file, dict)
        read_img = lambda x: img2tensor(x, bgr2rgb=True, float32=True) / 255.
        
        A, B = cv2.imread(self.keypose_path+file['primary']), cv2.imread(self.keypose_path+file['secondary'])
        # read
        A = deal(A)
        B = deal(B)
        # regular
        if self.flag == 0:
            w , h = A.shape[0] // self.factor, A.shape[1] // self.factor
            B = cv2.resize(B, (w,h), interpolation=Inter[self.inter])
            A = cv2.resize(A, (w,h), interpolation=Inter[self.inter])
            self.shape = (w, h)
            self.flag = 1
        elif self.flag==1:
            B = cv2.resize(B, self.shape, interpolation=Inter[self.inter])
            A = cv2.resize(A, self.shape, interpolation=Inter[self.inter])
        # B first
        # down sample and resize
        
        assert A.shape == B.shape, 'two keypose must have same shape: Shape1-{0}, Shape2-{1}'.format(A.shape, B.shape)
        
        if not A.shape == B.shape:
            B = rearrange(B, 'u v w -> v u w')
        prompt = file['prompt'].strip()
        # print('one group')
        
        assert A.shape==B.shape, 'here...'
        A = read_img(A)
        B = read_img(B)    
        assert A.shape==B.shape, 'here!!!'
        print(A.shape, B.shape)
        return {
            'primary': A,
            'secondary': B,
            'prompt': prompt
        }

    def __len__(self):
        return len(self.files)