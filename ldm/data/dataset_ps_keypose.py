# pretreatment of the (input, output) data

import json
import cv2
import pandas as pd
import os
from basicsr.utils import img2tensor
from random import randint, shuffle

class PsKeyposeDataset():
    def __init__(self, data_size, caption_path, keypose_path):
        # caption_path: csv file path -> read csv file
        # keypose_path: image folder -> store image names

        super(PsKeyposeDataset, self).__init__()
        try:
            prompts = list(pd.read_csv(caption_path)['CAPTION'])
        except:
            raise Exception('Read caption failed. Possible reason: \'CAPTION\' does not exists')

        listdir = os.listdir(keypose_path)
        length = len(listdir)
        assert length == len(prompts), 'Data preprocessing wrong.'
        index, image_list, prompt_list = [], [], []
        
        print('Data Gathering...')
        while len(index) <= data_size:
            one = randint(-1, length)
            if one in index:
                continue
            index.append(one)
            image_list.append(listdir[one])
            prompt_list.append(prompts[one])

        # image: OpenKeypose image

        self.files = []
        for A in range(len(index)):
            for B in range(0,A):
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
        read_img = lambda x: img2tensor(cv2.imread(x), bgr2rgb=True, float32=True) / 255.

        A, B = read_img(file['primary']), read_img(file['secondary'])
        prompt = file['prompot'].strip()

        return {
            'primray': A,
            'secondary': B,
            'prompt': prompt
        }

    def __len__(self):
        return len(self.files)