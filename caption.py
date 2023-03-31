import argparse
import os
import csv

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
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
    "--outdir",
    type=str,
    default='Datasets/Captions',
    help='output directions for captioning'
  )
  parser.add_argument(

  )


opt = parsr_args()


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = opt.max_length
num_beams = opt.num_beams
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def caption_step(opt):
  image_paths = opt.image
  images = [Image.open(image_paths if image_paths.mode == 'RGB' else image_paths.convert(mode='RGB'))] \
            if is_image_file(image_paths) \
          else [Image.open(one_image if one_image.mode == 'RGB' else one_image.convert(mode='RGB')) for one_image in os.listdir(image_paths) ]

  # integerate images

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [[pred.strip()] for pred in preds]
  preds.insert(0,['CAPTION'])
  output = '{opt.outdir}/{captions.csv}'
  with open(output, "w", newline="") as file:
      writer = csv.writer(file)
      for row in preds:
          writer.writerow(row)
  print("Images captioning done.")


# def getPose(opt):
#


caption_step(opt) # ['a woman in a hospital bed with a woman in a hospital bed']

