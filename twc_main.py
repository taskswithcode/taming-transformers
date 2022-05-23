import sys
import yaml
import torch
from omegaconf import OmegaConf
import pdb
from taming.models.vqgan import VQModel, GumbelVQ
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dall_e          import map_pixels, unmap_pixels, load_model
from IPython.display import display, display_markdown

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec



def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle: 
      img = map_pixels(img)
    return img


def reconstruct_with_dalle(x, encoder, decoder, do_preprocess=False):
  # takes in tensor (or optionally, a PIL image) and returns a PIL image
  if do_preprocess:
    x = preprocess(x)
  z_logits = encoder(x)
  z = torch.argmax(z_logits, axis=1)
  
  print(f"DALL-E: latent shape: {z.shape}")
  z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

  x_stats = decoder(z).float()
  x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
  x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

  return x_rec


def stack_reconstructions(input, x0, x1, x2, x3, titles=[]):
  font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)
  assert input.size == x1.size == x2.size == x3.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (5*w, h))
  img.paste(input, (0,0))
  img.paste(x0, (1*w,0))
  img.paste(x1, (2*w,0))
  img.paste(x2, (3*w,0))
  img.paste(x3, (4*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img

def reconstruction_pipeline(model1024, model16384, model32x32,encoder_dalle, decoder_dalle,url,device, size=320,is_local=False):
    titles=["Input", "DALL-E dVAE (f8, 8192)", "VQGAN (f8, 8192)", 
            "VQGAN (f16, 16384)", "VQGAN (f16, 1024)"]

    if (is_local):
        x_dalle = preprocess(PIL.Image.open(url), target_image_size=size, map_dalle=True)
        x_vqgan = preprocess(PIL.Image.open(url), target_image_size=size, map_dalle=False)
    else:
        x_dalle = preprocess(download_image(url), target_image_size=size, map_dalle=True)
        x_vqgan = preprocess(download_image(url), target_image_size=size, map_dalle=False)
    x_dalle = x_dalle.to(device)
    x_vqgan = x_vqgan.to(device)
    print(f"input is of size: {x_vqgan.shape}")
    x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model32x32)
    x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model16384)
    x2 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model1024)
    x3 = reconstruct_with_dalle(x_dalle, encoder_dalle, decoder_dalle)
    img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])), x3, 
                              custom_to_pil(x0[0]), custom_to_pil(x1[0]), 
                              custom_to_pil(x2[0]), titles=titles)
    return img

def load_models_and_compare(params):
    sys.path.append(".")

    # also disable grad to save memory
    torch.set_grad_enabled(False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")
    print("DEVICE is:",DEVICE)


    config1024 = load_config("logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    config16384 = load_config("logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)

    model1024 = load_vqgan(config1024, ckpt_path="logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)
    model16384 = load_vqgan(config16384, ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)


    config32x32 = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
    model32x32 = load_vqgan(config32x32, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True).to(DEVICE)


    encoder_dalle = load_model("./dalle_models/encoder.pkl", DEVICE)
    decoder_dalle = load_model("./dalle_models/decoder.pkl", DEVICE)




   


    #img = reconstruction_pipeline(url='https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1', size=384,is_local=False)
    img = reconstruction_pipeline(model1024, model16384, model32x32,encoder_dalle, decoder_dalle, url=params.input,device=DEVICE, size=384,is_local=params.is_local)
    print("Saving image into file:",params.output)
    img.save(params.output)
    print("Saving image into file: ",params.output," complete")
   # with open("ajit.txt","w") as fp:
   #     fp.write("This is from inside container\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQGan comparision with DALL-E DVae',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",required=True,help='Input image file name')
    parser.add_argument('-output', action="store", dest="output",required=True,help='Output image file name')
    parser.add_argument('-is_local', dest="is_local", action='store_true',help='Is Local')
    parser.add_argument('-no-is_local', dest="is_local", action='store_false',help='Is Remote')
    parser.set_defaults(is_local=True)
    results = parser.parse_args()
    load_models_and_compare(results)

