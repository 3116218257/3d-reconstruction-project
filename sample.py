# import torch
# from diffusers.utils import load_image
# from diffusers import StableDiffusionXLImg2ImgPipeline
# import PIL

# pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "./stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )

# device = f'cuda:{7}' if torch.cuda.is_available() else 'cpu'
# device = torch.device(device)
# pipe = pipe.to(device)

# init_image = PIL.Image.open('WechatIMG255.png')
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt, image=init_image).images

# image.save('new_image.jpg')
# url = "https://hf-mirror.com/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
# url = 'https://pica.zhimg.com/70/v2-7ff9a3358d9e1589d8f589f69b790b26_1440w.awebp?source=172ae18b&biz_tag=Post'
# init_image = load_image(url).convert("RGB")
# init_image.save("origin_pic.png")
# prompt = "let the man wear a football team uniform"
# image = pipe(prompt, image=init_image).images[0]
# image.save("astronaut_rides_horse.png")


# git clone https://hf-mirror.com/stabilityai/stable-diffusion-xl-refiner-1.0
#https://hf-mirror.com/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors
#https://hf-mirror.com/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/text_encoder_2/model.safetensors
#https://hf-mirror.com/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors

#https://hf-mirror.com/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae/diffusion_pytorch_model.safetensors
#https://hf-mirror.com/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/vae_1_0/diffusion_pytorch_model.safetensors


import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import PIL
import cv2

model_id = "./instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
pipe = pipe.to(device)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
image = PIL.Image.open('people1.jpg')
# `image` is an RGB PIL.Image
images = pipe("let him wear a soccer uniform", image=image).images
images[0].save('processed_image_baseball.png')

# import cv2
# import numpy as np
# from torchvision import transforms

# def image_segmentation(image_path):
#     image = cv2.imread(image_path)
#     mask = np.zeros(image.shape[:2], np.uint8)

#     bgd_model = np.zeros((1, 65), np.float64)
#     fgd_model = np.zeros((1, 65), np.float64)
    
#     rectangle = (50, 50, image.shape[1]-50, image.shape[0]-50)
    
#     cv2.grabCut(image, mask, rectangle, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#     result = image * mask2[:, :, np.newaxis]
    
#     return result

# image_path = 'origin_pic.png' 
# segmented_image = image_segmentation(image_path)

# tmp = segmented_image
# image = transforms.ToPILImage()(tmp)
# image.save(f"segmentaed_image.png")

