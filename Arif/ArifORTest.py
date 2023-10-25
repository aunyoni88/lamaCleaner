from pathlib import Path
import cv2
import torch
import contextlib
import os
import numpy as np
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler
from lama_cleaner.model_manager import ModelManager

current_dir = Path(__file__).parent.absolute().resolve()
image_dir = current_dir / "image"
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
device = torch.device(device)
print(f"arif device == {device}")
imageName = "16"
sampler = SDSampler.ddim
prompt = "An indian women in red sari with wings"
n_prompt = ""
sd_steps = 100



def get_data(
        fx: float = 1.0,
        fy: float = 1.0,
        img_p=image_dir / f"{imageName}.jpg",
        mask_p=image_dir / f"{imageName}.1.jpg",
):
    img = cv2.imread(str(img_p))
    print(img.size)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    height = img.shape[0]
    width = img.shape[1]

    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

    return img, mask, width, height


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def show_image(output_img, width, height):
    img = cv2.cvtColor(output_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # res_np_img = image_resize(img, height=800)

    dim = (width, height)
    resized = cv2.resize(output_img, dim, interpolation=cv2.INTER_AREA)

    filename = 'result.jpg'
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)

    cv2.imwrite(filename, resized)


def run_model(model_name):
    model = ModelManager(
        name=model_name,
        device=device,
        hf_access_token="",
        sd_run_local=False,
        disable_nsfw=False,
        sd_cpu_textencoder=False,
    )

    config = Config(ldm_steps=1,
                    ldm_sampler=LDMSampler.plms,
                    hd_strategy=HDStrategy.ORIGINAL,
                    sd_steps=sd_steps,
                    prompt=prompt,
                    negative_prompt=n_prompt,
                    sd_sampler=sampler,
                    sd_match_histograms=True,
                    hd_strategy_crop_margin=32,
                    hd_strategy_crop_trigger_size=200,
                    hd_strategy_resize_limit=200)
    img, mask, width, height = get_data()
    output_img = model(img, mask, config)
    show_image(output_img, width, height)
def run_realistic():
    model_n = "realisticVision1.4"
    run_model(model_name=model_n)

def run_anything():
    model_n = "anything4"
    run_model(model_name=model_n)


run_realistic()
# run_anything()
