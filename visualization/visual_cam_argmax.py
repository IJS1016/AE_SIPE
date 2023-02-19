import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5"

import argparse
from PIL import Image
from tool import pyutils
from data import data_voc, data_coco
from tqdm import tqdm
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import importlib
import imageio

"""
Visualization CAM argmax map from cam numpy files
"""

VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                        (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)

image_path = "/home/jsim/Desktop/datasets/voc2012_seg/trainaug_all_images/"

imageio.imsave(cam_viz_path, heatmap)

def draw_segmap(seg_map, orig_img, save_path, img_name):
    seg_map = np.array(Image.fromarray((seg_map * 255).astype(np.uint8), 'RGB').resize((orig_img.shape[1], orig_img.shape[0]), Image.BICUBIC))

    if seg_map.shape == orig_img.shape:
        out = (seg_map + np.array(orig_img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(seg_map.shape)
        print(np.array(orig_img).shape)

    cam_viz_path = os.path.join(save_path, img_name + '.png')
    imageio.imsave(cam_viz_path, out)

    print(f"saved {img_name}")

def run(args, predict_dir, num_cls, session_path):
    preds = []
    masks = []
    n_images = 0

    save_path = f"{session_path}/cam_semseg_color"
    os.makedirs(save_path, exist_ok=True)

    for iter, pack in tqdm(enumerate(dataloader)):
        n_images += 1
        cam_dict = np.load(os.path.join(predict_dir, pack['name'][0] + '.npy'), allow_pickle=True).item()
        cams = cam_dict['IS_CAM']
        keys = cam_dict['keys']
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

        mask = np.array(Image.open(os.path.join(args.gt_path,  pack['name'][0] + '.png')))
        masks.append(mask.copy())

        # orig_img = np.array(Image.open(f"{image_path}/{pack['name'][0]}.jpg").convert('RGB'))
        # draw_segmap(VOC_color[cls_labels], orig_img, save_path, pack['name'][0])
        cam_viz_path = os.path.join(save_path, pack['name'][0] + '.png')
        imageio.imsave(cam_viz_path, VOC_color[cls_labels])

    confusion = calc_semantic_segmentation_confusion(preds, masks)[:num_cls, :num_cls]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print({'iou': iou, 'miou': np.nanmean(iou)})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--gt_path", default='../PascalVOC2012/VOCdevkit/VOC2012/SegmentationClass', type=str)
    parser.add_argument('--session_name', default="exp", type=str)
    args = parser.parse_args()

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '../PascalVOC2012/VOCdevkit/VOC2012'
        num_cls = 21
        dataset = data_voc.VOC12ImageDataset('data/train_' + args.dataset + '.txt', voc12_root=dataset_root, img_normal=None, to_torch=False)

    elif args.dataset == 'coco':
        args.gt_path = "../ms_coco_14&15/SegmentationClass/train2014/"
        dataset_root = "../ms_coco_14&15/images"
        num_cls = 81
        dataset = data_coco.COCOImageDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root, img_normal=None, to_torch=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    pyutils.Logger(os.path.join(args.session_name, 'eval_' + args.session_name + '.log'))
    run(args, args.session_name + "/npy", num_cls, args.session_name)