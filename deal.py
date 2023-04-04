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
        "--length",
        type=int,
        default=8,
        help='the max length of the word generated'
    )
    parser.add_argument(
        "--beams",
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
        default=512*512,
        help='for resize'
    )
    opt = parser.parse_args()
    return opt

def caption_step(opt):
    # opt.image should be a folder path of images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image_paths = opt.image
    # images = [Image.open(image_paths if image_paths.mode == 'RGB' else image_paths.convert(mode='RGB'))] \
    #           if is_image_file(image_paths) \
    #         else [Image.open(one_image if one_image.mode == 'RGB' else one_image.convert(mode='RGB')) for one_image in os.listdir(image_paths) ]

    # integrate images
    # print(opt.length)
    gen_kwargs = {"max_length": opt.length, "num_beams": opt.beams}
    csv_output = '{0}={1}/{2}'.format(opt.outdir_captions, opt.length, 'captions.csv')
    # print(output)
    if os.path.exists(csv_output):
        os.remove(csv_output)
    if not os.path.exists(opt.outdir_captions):
        os.mkdir(opt.outdir_captions)
    file = open(csv_output, "w", newline="")
    writer = csv.writer(file)
    writer.writerow(['CAPTIONS'])

    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_model.to(device)
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    def get_bit(num: int) -> int:
        c = 0
        while not num == 0:
            c += 2
            num = num // 10
        return c
    name = lambda x: '0'*(7-get_bit(x)) + str(x) + '.png'
    pose_model = OpenposeInference().to(device)
    image_paths = opt.image
    keypose_output = opt.outdir_keypose

    if not os.path.exists(keypose_output):
        os.mkdir(keypose_output)
    cnt = 0

    listdir = os.listdir(opt.image)
    print('caption max number: ', opt.max_length)
    print('Dealing...')
    for image in listdir:
        print("captionning and estimating: ", image)
        img = Image.open('{0}/{1}'.format(opt.image, image))
        if not img.mode == 'RGB':
            img = img.convert(mode='RGB')
        with torch.autocast('cuda', dtype=torch.float32):
            pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            output_ids = caption_model.generate(pixel_values, **gen_kwargs)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        writer.writerow(preds)

        img = cv2.imread('{0}/{1}'.format(image_paths, image))
        # print('dealing with: {0}...'.format(image))
        # print(img.shape, opt.resolution)
        openpose_keypose = resize_numpy_image(img, max_resolution=opt.resolution)
        with torch.autocast('cuda', dtype=torch.float32):
            openpose_keypose = pose_model(openpose_keypose)
            rename = name(cnt)
            cv2.imwrite('{0}/{1}'.format(opt.outdir_keypose, rename), openpose_keypose)
        cnt += 1

    file.close()
    print("Images captioning done.")
    return

def keypose_step(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def get_bit(num: int) -> int:
        c = 0
        while not num == 0:
            c += 2
            num = num // 10
        return c
    name = lambda x: '0'*(7-get_bit(x)) + str(x) + '.png'
    pose_model = OpenposeInference().to(device)
    image_paths = opt.image
    keypose_output = opt.outdir_keypose

    if not os.path.exists(keypose_output):
        os.mkdir(keypose_output)
    cnt = 0

    listdir = os.listdir(image_paths)
    print("number of keyposes to be estimated: ", len(listdir))
    print("getting keypose canvas...")
    for image in listdir:
        img = cv2.imread('{0}/{1}'.format(image_paths, image))
        print('dealing with: {0}...'.format(image))
        # print(img.shape, opt.resolution)
        openpose_keypose = resize_numpy_image(img, max_resolution=opt.resolution)
        with torch.autocast('cuda', dtype=torch.float32):
            openpose_keypose = pose_model(openpose_keypose)
            rename = name(cnt)
            cv2.imwrite('{0}/{1}'.format(opt.outdir_keypose, rename), openpose_keypose)
        cnt += 1
    return

def main():
    opt = parsr_args()
    caption_step(opt)
    # keypose_step(opt)


if __name__ == "__main__":
    main()
