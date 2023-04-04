import os
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="/root/Ps-Adapter/Datasets/Data",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/root/Ps-Adapter/Datasets/sets"
    )
    opt = parser.parse_args()
    return opt

def get_bit(num):
    u = 0
    while num!=0:
        u += 1
        num = num // 10
    return u

def get_name(num):
    t = get_bit(num)
    assert t <= 7, 'The amount of total images exceeds.'
    return ('0'*(7-t)+str(num)+".png")

def ends(f: str):
    return True if f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png') else False
        

def get_files(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if ends(file_path.lower()):
                file_list.append(file_path)
    # print(file_list)
    return file_list



def main():
    opt = parser_args()
    PATH = opt.source
    path = opt.target
    file_list = get_files(path)

    cnt = 0
    from shutil import move as mv
    from os import rename as rn

    if not os.path.exists(PATH):
        os.mkdir(PATH)

    for file_path in file_list:
        root_list = file_path.split('/')
        root_list[-1] = get_name(cnt)
        cnt += 1
        new_name = '/'.join(root_list)
        # print(file_path, ' -> ', new_name)
        # print(new_name)
        rn(file_path, new_name)
        mv(new_name, PATH)
        if cnt % 10000 == 1:
            print("Still Running at: cnt = ", cnt)


if __name__=="__main__":
    main()