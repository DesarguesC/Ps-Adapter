import os
cnt = 0
PATH = '/root/Ps-Adapter/Data/'     #target path 

def get_bit(num):
    cnt = 0
    while num!=0:
        cnt += 1
        num = num // 10
    return cnt

def get_name(num):
    t = get_bit(num)
    assert t <= 7, 'The amount of total images exceeds.'
    return ('0'*(7-t)+str(num)+".png")

def ends(f: str):
    return f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('png')
        

def get_files(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if ends(file_path.lower()):
                file_list.append(file_path)
    return file_list

path = "/root/Ps-Adapter/Datasets"  # original images storage path
file_list = get_files(path)

from shutil import move as mv
from os import rename as rn

for file_path in file_list:
    root_list = file_path.split('/')
    root_list[-1] = get_name(cnt)
    cnt += 1
    new_name = '/'.join(root_list)
    rn(new_name, file_path)
    mv(new_name, PATH)

