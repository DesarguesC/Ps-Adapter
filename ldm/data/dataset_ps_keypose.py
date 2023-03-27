# pretreatment of the (input, output) data

import json
import cv2
import os
from basicsr.utils import img2tensor

class PsKeyposeDataset():
    def __init__(self, file_path):
        super(PsKeyposeDataset, self).__init__()

        self.files = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # 一个文件中每一行都是条数据五元组中的文件路径，中间用空格分隔

            for line in lines:
                path_list = line.strip()
                assert len(path_list)==5
                keypose_path_1, keypose_path_2 = \
                    path_list[0], path_list[1]
                txt_path = path_list[2]
                img_path_1, img_path_2 = \
                    path_list[3], path_list[4]
                self.files.append({
                    "keypose_path_1": keypose_path_1,   # image path
                    "keypose_path_2": keypose_path_2,   # image path
                    "txt_path": txt_path,                   # txt path
                    "img_path_1": img_path_1,           # jpg/jpeg/png
                    "img_path_2": img_path_2            # jpg/jpeg/png
                     })

        # 文件中都用txt存图片路径，prompt直接读


    def __getitem__(self, idx):
        file = self.files[idx]
        assert isinstance(file, dict)
        read_img = lambda x: img2tensor(cv2.imread(x), bgr2rgb=True, float32=True) / 255.

        img_1, img_2 = read_img(file['img_path_1']), read_img(file['img_path_2'])
        keypose_1, keypose_2 = read_img(file['keypose_path_1']), read_img(file['keypose_path_2'])
        # keypose点构建灰度图读入？

        with open(file["txt_path"]) as p:
            prompt = p.readline().strip()

        return {
            "img_1": img_1,
            "img_2": img_2,
            "prompt": prompt,
            "keypose_1": keypose_1,
            "keypose_2": keypose_2
        }

    def __len__(self):
        return len(self.files)