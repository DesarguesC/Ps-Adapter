import argparse
import os
import csv
import cv2
from ldm.modules.extra_condition.openpose.api import OpenposeInference
from ldm.util import resize_numpy_image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from basicsr.utils import img2tensor
from PIL import Image


def is_image_file(image_path: str) -> bool:
    return image_path.lower().endswith('.jpeg') or image_path.lower().endswith('.jpg') \
         or image_path.lower().endswith('.png') or image_path.lower().endswith('.webp')

def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_length",
        type=int,
        default=8,
        help='the max length of the word generated'
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help='embedding nums'
    )
    parser.add_argument(
        "--image",
        type=str,
        default='Datasets/Data',
        help='image path / image folder'
    )
    parser.add_argument(
        "--outdir_captions",
        type=str,
        default='Datasets/Captions',
        help='output directions for captioning'
    )
    parser.add_argument(
        "--outdir_keypose",
        type=str,
        default='Datasets/Keypose',
        help='output directions for keypose'
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help='for resize'
    )

def caption_step(opt, **gen_kargs):
    # opt.image should be a folder path of images


    # image_paths = opt.image
    # images = [Image.open(image_paths if image_paths.mode == 'RGB' else image_paths.convert(mode='RGB'))] \
    #           if is_image_file(image_paths) \
    #         else [Image.open(one_image if one_image.mode == 'RGB' else one_image.convert(mode='RGB')) for one_image in os.listdir(image_paths) ]

    # integerate images

    output = '{opt.outdir_captions}/{captions.csv}'
    with open(output, "w", newline="") as file:
        writer = csv.writer(file)
    writer.writerow(['CAPTIONS'])

    for image in os.listdir(opt.image):
        img = Image.open(image)
        if not img.mode == 'RGB':
            img = img.convert(mode='RGB')
        with torch.autocast('cuda', dtype=torch.float32):
            pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            output_ids = caption_model.generate(pixel_values, **gen_kwargs)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        writer.writerow(preds)

    file.close()
    print("Images captioning done.")
    return

def keypose_step(opt):
    def get_bit(num: int) -> int:
        c = 0
        while not num==0:
            c += 2
            num = num // 10
        return c
    name = lambda x: '0'*(8-x) + str(get_bit(x)) + '.png'
    pose_model = OpenposeInference().to(device)
    image_paths = opt.image

    # images = [Image.open(image_paths if image_paths.mode == 'RGB' else image_paths.convert(mode='RGB'))] \
    #   if is_image_file(image_paths) \
    #   else [Image.open(one_image if one_image.mode == 'RGB' else one_image.convert(mode='RGB')) for one_image in
    #         os.listdir(image_paths)]

    cnt = 0
    for image in os.listdir(image_paths):
        img = cv2.imread(image)
        openpose_keypose = resize_numpy_image(img, max_resolution=opt.resolution)
        # openpose_keypose.shape[:2]
        with torch.autocast('cuda', dtype=torch.float32):
            openpose_keypose = pose_model(openpose_keypose)
        openpose_keypose = img2tensor(openpose_keypose).unsqueeze(0)
        openpose_keypose = openpose_keypose.to('cpu')
        rename = name(cnt)
        cv2.imwrite(openpose_keypose, '{opt.outdir_keypose}/{rename}')
    return

opt = parsr_args()

caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)

max_length = opt.max_length
num_beams = opt.num_beams
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

caption_step(opt, gen_kwargs)
keypose_step(opt)
