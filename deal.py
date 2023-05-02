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


Inter = {
    'inter_cubic': cv2.INTER_CUBIC,
    'inter_linear': cv2.INTER_LINEAR,
    'inter_nearest': cv2.INTER_NEAREST,
    'inter_lanczos4': cv2.INTER_LANCZOS4
}

def is_image_file(image_path: str) -> bool:
    return image_path.lower().endswith('.jpeg') or image_path.lower().endswith('.jpg') \
         or image_path.lower().endswith('.png') or image_path.lower().endswith('.webp')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--length",
        type=int,
        default=30,
        help='the max length of the word generated'
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=5,
        help='embedding nums'
    )
    parser.add_argument(
        "--caption_model",
        type=str,
        default='vit-gpt2',
        choices=['vit-gpt2'],
        help='which image caption model to be used'
    )
    parser.add_argument(
        "--imcp_path",
        type=str,
        default="nlpconnect/vit-gpt2-image-captioning",
        help='model basic path of image captioning model (vit-gpt2-image-captioning)'
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
    parser.add_argument(
        "--random_num",
        type=int,
        default=1800,
        help='choose 1800 samples'
    )
    parser.add_argument(
        "--resize",
        type=str2bool,
        default=True,
        help='ensure images the same shape'
    )
    parser.add_argument(
        "--inter",
        type=str,
        default='inter_cubic',
        choices=['inter_cubic', 'inter_linear', 'inter_nearest', 'inter_lanczos4'],
        help='resize shape'
    )
    opt = parser.parse_args()
    return opt

def caption_step(opt):
    # opt.image should be a folder path of images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt.outdir_captions = opt.outdir_captions if opt.outdir_captions.endswith('/') else opt.outdir_captions+'/'
    opt.outdir_keypose = opt.outdir_keypose if opt.outdir_keypose.endswith('/') else opt.outdir_keypose+'/'
    opt.image = opt.image if opt.image.endswith('/') else opt.image + '/'

    gen_kwargs = {"max_length": opt.length, "num_beams": opt.beams}
    csv_output = opt.outdir_captions + 'captions.csv'

    if os.path.exists(csv_output):
        os.remove(csv_output)
    if not os.path.exists(opt.outdir_captions):
        print('bug')
        os.mkdir(opt.outdir_captions)
    file = open(csv_output, "w", newline="")
    writer = csv.writer(file)
    writer.writerow(['CAPTIONS'])

    version = opt.imcp_path
    caption_model = VisionEncoderDecoderModel.from_pretrained(version)
    caption_model.to(device)
    feature_extractor = ViTImageProcessor.from_pretrained(version)
    tokenizer = AutoTokenizer.from_pretrained(version)

    # rename images
    def get_bit(num: int) -> int:
        c = 0
        while not num == 0:
            c += 1
            num = num // 10
        return c
    name = lambda x: '0'*(6-get_bit(x)) + str(x) + '.png'

    pose_model = OpenposeInference().to(device)
    image_paths = opt.image
    keypose_output = opt.outdir_keypose

    if not os.path.exists(keypose_output):
        os.mkdir(keypose_output)
    cnt = 0

    index, listdir = [], []
    lists = os.listdir(image_paths)

    print('Data Choosing...')
    from random import randint
    while len(index) <= opt.random_num:
        one = randint(-1, opt.random_num)
        if one in index:
            continue
        index.append(one)
        if lists[one].endswith('.jpg'):
            listdir.append(lists[one])

    print('caption max number: ', opt.length)
    print('Dealing...')

    for image in listdir:
        # image in name list; image ->
        print("captionning and estimating: ", image)
        img = Image.open(image_paths + image)
        if not img.mode == 'RGB':
            img = img.convert(mode='RGB')
        # image captioning
        with torch.autocast('cuda', dtype=torch.float32):
            pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            output_ids = caption_model.generate(pixel_values, **gen_kwargs)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        writer.writerow(preds)

        img = cv2.imread(image_paths + image)
        openpose_keypose = resize_numpy_image(img, max_resolution=opt.resolution, resize_method=Inter[opt.inter])
        with torch.autocast('cuda', dtype=torch.float32):
            openpose_keypose = pose_model(openpose_keypose)
            rename = name(cnt)
            cv2.imwrite(keypose_output + rename, openpose_keypose)
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
        img = cv2.resize(img, (img.shape[:2] / opt.factor), interpolation= Inter[opt.inter])
        print('dealing with: {0}...'.format(image))
        # print(img.shape, opt.resolution)
        openpose_keypose = resize_numpy_image(img, max_resolution=opt.resolution)
        with torch.autocast('cuda', dtype=torch.float32):
            openpose_keypose = pose_model(openpose_keypose)
            rename = name(cnt)
            cv2.imwrite('{0}/{1}'.format(opt.outdir_keypose, rename), openpose_keypose)
        cnt += 1
    return

def debug(opt):
    print(opt.shape)
    import sys
    sys.exit(0)

def main():
    opt = parser_args()
    # debug(opt)
    caption_step(opt)
    # keypose_step(opt)


if __name__ == "__main__":
    main()
