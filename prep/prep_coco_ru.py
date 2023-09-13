import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
from tqdm import tqdm
import argparse

import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from rudalle import get_vae


def preprocess(img, target_image_size):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img


def encode(vq, x):
    return vq.model.encode((2 * x - 1))[-1][-1]


class PrepCoCoDS(Dataset):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        image_name = self.data[item]['file_name']
        image_path = f"%s/%s/%s" % (self.coco_dir, self.suffix, image_name)
        is_error = False
        image_d2 = self.dummy_d2
        image_d3 = self.dummy_d3
        try:
            image_d2 = preprocess(Image.open(image_path).convert("RGB"), 128)
            image_d3 = preprocess(Image.open(image_path).convert("RGB"), 256)
        except PIL.UnidentifiedImageError:
            is_error = True
        except OSError:
            is_error = True
        except BaseException:
            is_error = True
        except:
            is_error = True
        if is_error:
            return image_d2, image_d3, "", image_name
        return image_d2, image_d3, "success", image_name

    def __init__(self, data, coco_dir, suffix: str):
        if suffix == 'train':
            self.suffix = 'train2014'
        else:
            self.suffix = 'val2014'
        self.data = data
        self.coco_dir = coco_dir
        self.dummy_d2 = torch.zeros(1, 3, 128, 128)
        self.dummy_d3 = torch.zeros(1, 3, 256, 256)


def main(bs, coco_dir, coco_ann_dir, out_dir):
    device = torch.device('cuda:0')
    vqmodel = get_vae().to(device)
    vqmodel.eval().requires_grad_(False)

    with open('%s/captions_train2014.json' % coco_ann_dir, 'r') as f:
        data = json.load(f)
    train_data = data["images"]
    print("%0d captions loaded from json " % len(train_data))
    with open('%s/captions_val2014.json' % coco_ann_dir, 'r') as f:
        data = json.load(f)
    test_data = data["images"]
    print("%0d captions loaded from json " % len(test_data))
    for suffix in ("val", "train"):
        counter = 0
        if suffix == "train":
            data = train_data
        else:
            data = test_data

        ds = PrepCoCoDS(data, coco_dir, suffix)
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, drop_last=False)
        progress = tqdm(total=len(dl))
        for i, data in enumerate(dl):
            images_d2, images_d3, captions, image_names = data
            images_d2 = images_d2.to(device)
            images_d3 = images_d3.to(device)
            with torch.no_grad():
                z2 = encode(vqmodel, images_d2.squeeze(1)).cpu()
                z3 = encode(vqmodel, images_d3.squeeze(1)).cpu()

            is_valid = list(map(lambda x: x != "", captions))
            mask = torch.tensor(is_valid)
            tokens2 = z2[mask]
            tokens3 = z3[mask]
            images_32 = images_d3.cpu()[mask]
            image_names = [image_name for j, image_name in enumerate(image_names) if is_valid[j]]
            for j in range(len(image_names)):
                indice = torch.tensor([j])
                tokens2_0 = torch.index_select(tokens2, 0, indice)
                tokens3_0 = torch.index_select(tokens3, 0, indice)
                images32_0 = torch.index_select(images_32, 0, indice)
                img_id = image_names[j].split('_')[-1].split('.')[0][-6:]
                out_data_dir = os.path.join(out_dir, suffix)
                if not os.path.exists(out_data_path):
                    os.makedirs(out_data_dir)
                out_data_path = '%s/coco_ViT-B_%s_%012d.pt' % (out_data_dir, suffix, img_id)
                torch.save({"vtokens_16": tokens2_0, "vtokens_32": tokens3_0,
                            "images_32": images32_0,
                            "image_name": image_names[j]},
                           out_data_path)
            progress.update()
            counter += len(image_names)

        progress.close()
        print('Done')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--coco_dir', type=str, default="./coco")
    parser.add_argument('--coco_ann_dir', type=str, default="./annotations")
    parser.add_argument('--out_dir', type=str, default="./prep_coco_ru")
    args = parser.parse_args()
    exit(main(args.bs, args.coco_dir, args.coco_ann_dir, args.out_dir))
