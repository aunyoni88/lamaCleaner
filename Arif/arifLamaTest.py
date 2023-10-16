from pathlib import Path
import cv2
import torch
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
from lama_cleaner.model_manager import ModelManager

current_dir = Path(__file__).parent.absolute().resolve()
image_dir = current_dir / "image"
save_dir = current_dir / "result"
save_dir.mkdir(exist_ok=True, parents=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
device = torch.device(device)
print(f"arif device == {device}")
imageName = "4"

def get_data(
        fx: float = 1,
        fy: float = 1.0,
        img_p=image_dir / f"{imageName}.jpg",
        mask_p=image_dir / f"{imageName}.1.jpg",
):
    img = cv2.imread(str(img_p))
    print(img.size)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    return img, mask


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


def show_image(output_img):
    resized_image = image_resize(output_img, height=800)
    window_name = 'Output Image'
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_lama():
    model = ModelManager(name="lama", device=device)
    config = Config(ldm_steps=1,
                    ldm_sampler=LDMSampler.plms,
                    hd_strategy=HDStrategy.CROP,
                    hd_strategy_crop_margin=32,
                    hd_strategy_crop_trigger_size=200,
                    hd_strategy_resize_limit=200)
    fx: float = 1.3
    fy: float = 1.0
    img, mask = get_data(fx=fx, fy=fy)
    output_img = model(img, mask, config)
    show_image(output_img)


def run_zits():
    model = ModelManager(name="zits", device=device)
    config = Config(ldm_steps=1,
                    ldm_sampler=LDMSampler.plms,
                    hd_strategy=HDStrategy.CROP,
                    hd_strategy_crop_margin=32,
                    hd_strategy_crop_trigger_size=200,
                    hd_strategy_resize_limit=200,
                    zits_wireframe=True)

    fx: float = 1.3
    fy: float = 1.0
    img, mask = get_data(fx=fx, fy=fy)
    output_img = model(img, mask, config)
    show_image(output_img)


def run_fcf():
    model = ModelManager(name="fcf", device=device)
    config = Config(ldm_steps=1,
                    ldm_sampler=LDMSampler.plms,
                    hd_strategy=HDStrategy.CROP,
                    hd_strategy_crop_margin=32,
                    hd_strategy_crop_trigger_size=200,
                    hd_strategy_resize_limit=200)

    fx: float = 2.0
    fy: float = 2.0
    img, mask = get_data(fx=fx, fy=fy)
    output_img = model(img, mask, config)
    show_image(output_img)


def run_mat():
    no_half = True
    model = ModelManager(name="mat", device=device, no_half=no_half)
    config = Config(ldm_steps=1,
                    ldm_sampler=LDMSampler.plms,
                    hd_strategy=HDStrategy.CROP,
                    hd_strategy_crop_margin=32,
                    hd_strategy_crop_trigger_size=200,
                    hd_strategy_resize_limit=200)

    fx: float = 1.0
    fy: float = 1.0
    img, mask = get_data(fx=fx, fy=fy)
    output_img = model(img, mask, config)
    show_image(output_img)


def run_ldm():
    model = ModelManager(name="ldm", device=device)
    config = Config(ldm_steps=10,
                    ldm_sampler=LDMSampler.plms,
                    hd_strategy=HDStrategy.ORIGINAL,
                    hd_strategy_crop_margin=32,
                    hd_strategy_crop_trigger_size=200,
                    hd_strategy_resize_limit=200)

    fx: float = 1.3
    fy: float = 1.0
    img, mask = get_data(fx=fx, fy=fy)
    output_img = model(img, mask, config)
    show_image(output_img)


# run_lama()
run_zits()
# run_fcf()
# run_mat()
# run_ldm()
