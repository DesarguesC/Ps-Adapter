# pretreatment of the (input, output) data

import json
import cv2
import pandas as pd
import os
from basicsr.utils import img2tensor
from random import randint, shuffle
from ldm.util import resize_numpy_image

class PsKeyposeDataset():
    def __init__(self, caption_path, keypose_path, resize=False, max_resolution=512*512):
        # caption_path: csv file path -> read csv file
        # keypose_path: image folder -> store image names
        self.caption_path = caption_path
        self.keypose_path = keypose_path if keypose_path.endswith('/') else keypose_path + '/'
        self.resize = resize
        self.max_resolution = max_resolution

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
        read_img = lambda x: img2tensor(resize_numpy_image(cv2.imread(x), max_resolution=self.max_resolution) \
                                            if self.resize else cv2.imread(x), bgr2rgb=True, float32=True) / 255.
        
        A, B = read_img(self.keypose_path+file['primary']), read_img(self.keypose_path+file['secondary'])
        assert A.shape == B.shape, 'two keypose must have same shape: Shape1-{0}, Shape2-{1}'.format(A.shape, B.shape)
        prompt = file['prompt'].strip()
        print(A.shape, B.shape)
        
        return {
            'primary': A,
            'secondary': B,
            'prompt': prompt
        }

    def __len__(self):
        return len(self.files)