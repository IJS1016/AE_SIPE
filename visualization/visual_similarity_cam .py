
import os
from builtins import bool
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import importlib

import cv2
import imageio
import argparse
from data import data_voc, data_coco
from tool import torchutils, pyutils

cudnn.enabled = True

# for cam visualization ##################################################
def overlap(img, hm):
    hm = hm.squeeze()

    hm = plt.cm.jet(hm)[:, :, :3]
    hm = np.array(Image.fromarray((hm*255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(np.float)*2
    if hm.shape == np.array(img).astype(np.float).shape:
        out = (hm + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(hm.shape)
        print(np.array(img).shape)
    return out

def draw_heatmap(norm_cam, keys, orig_img, save_path, img_name):
    for gt in range(len(norm_cam)):
        heatmap = overlap(orig_img, norm_cam[gt])
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(int(keys[gt])))
        imageio.imsave(cam_viz_path, heatmap)

        # print(f"finish {cam_viz_path}")
##########################################################################


# for seedregion visualization ##################################################
def draw_iscam(seed_region, orig_img, save_path, img_name, keys) :
    seed_region = seed_region.squeeze()
    seed_region = np.array(seed_region * 255).astype(np.uint8)

    for i, k in enumerate(keys) :
        sr = seed_region[i]
        sr_color = cv2.cvtColor(sr, cv2.COLOR_GRAY2RGB)
        sr_color[sr == 255] = [255, 0, 0] # set to red

        cls_seed_region = np.array(Image.fromarray(sr_color, 'RGB'))

        out = (cls_seed_region * 2 + np.array(orig_img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)

        cam_viz_path = f"{save_path}/{img_name}_{k}.png"
        imageio.imsave(cam_viz_path, out)

def save_var_iscam(cam, hie_fea, orig_img, img_name, keys, save_path) :
    hie_fea = hie_fea.to('cpu')

    # make var_segmap
    prob_sum = cam.sum(0)
    cam_ratio = cam / prob_sum

    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # negative
    for th in ths : 
        _save_path = f"{save_path}/{th}"
        os.makedirs(_save_path, exist_ok=True)

        seed_region = cam_ratio > th
        seed_region = seed_region.int()
        seed_region = seed_region.unsqueeze(0)

        prototypes = get_prototype(seed_region, hie_fea)
        IS_cam = reactivate(prototypes, hie_fea)

        IS_cam /= F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5
        IS_cam = IS_cam.cpu().numpy()
        
        draw_heatmap(IS_cam, keys, orig_img, save_path, img_name)

    # adequate
    argmax_cam = cam.argmax(0)
    ade_seed_region = torch.zeros_like(cam)
    
    for c in range(cam.shape[0]) :
        ade_seed_region[c, :, :] = (argmax_cam == c)

    _save_path = f"{save_path}/argmax"
    os.makedirs(_save_path, exist_ok=True)

    draw_iscam(ade_seed_region, orig_img, _save_path, img_name, keys)

def draw_seedregion(seed_region, orig_img, save_path, img_name, keys) :
    seed_region = seed_region.squeeze()
    seed_region = np.array(seed_region * 255).astype(np.uint8)

    for i, k in enumerate(keys) :
        sr = seed_region[i]
        sr_color = cv2.cvtColor(sr, cv2.COLOR_GRAY2RGB)
        sr_color[sr == 255] = [255, 0, 0] # set to red

        cls_seed_region = np.array(Image.fromarray(sr_color, 'RGB'))

        out = (cls_seed_region * 2 + np.array(orig_img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)

        cam_viz_path = f"{save_path}/{img_name}_{k}.png"
        imageio.imsave(cam_viz_path, out)

#############################################################################

def draw_segmap(seg_map, orig_img, save_path, img_name):
    seg_map = seg_map.squeeze()
    seg_map = np.array(Image.fromarray((seg_map * 255).astype(np.uint8), 'RGB').resize((orig_img.shape[1], orig_img.shape[0]), Image.BICUBIC))

    if seg_map.shape == orig_img.shape:
        out = (seg_map + np.array(orig_img / 4).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(seg_map.shape)
        print(np.array(orig_img).shape)

    cam_viz_path = os.path.join(save_path, img_name + '.png')
    imageio.imsave(cam_viz_path, out)

    print(f"saved {img_name}")


def denormalization(image) :
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    
    if image.dim() == 3:
        assert image.dim() == 3, "Expected image [CxHxW]"
        assert image.size(0) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)

    elif image.dim() == 4:
        # batch mode
        assert image.size(1) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip((0,1,2), mean, std):
            image[:, t, :, :].mul_(s).add_(m)

    return image

def _work(process_id, model, dataset, args):
    save_path = f"{args.session_name}/intermediate_results/"
    os.makedirs(save_path, exist_ok=True)

    save_list = ['cam', 'seeds1', 'IScam',  'seeds2', 'refined_IScam']
    for s in save_list :
        os.makedirs(f"{save_path}/{s}", exist_ok=True)

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    os.makedirs(os.path.join(args.session_name, "visual_per_size"), exist_ok=True)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        model.training = False

        for iter, pack in enumerate(data_loader):
            if iter > 50 : 
                break
            
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            label = F.pad(label, (1, 0), 'constant', 1.0)

            keys = torch.nonzero(label).flatten()

            orig_img = np.array(Image.open(f"{pack['img_path'][0]}").convert('RGB'))

            # original
            norm_cam, IS_cam, IS_cam2, seeds1, seeds2 = model(pack['img'][0].cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1), denormalization(pack['img'][0]).cuda(non_blocking=True))

            norm_cam = F.interpolate(norm_cam, size=size, mode='bilinear')[0]
            IS_cam = F.interpolate(IS_cam, size=size, mode='bilinear')[0]
            IS_cam2 = F.interpolate(IS_cam2, size=size, mode='bilinear')[0]
            seeds1 = F.interpolate(seeds1, size=size, mode='bilinear')[0]
            seeds2 = F.interpolate(seeds2, size=size, mode='bilinear')[0]

            norm_cam = norm_cam[keys]
            IS_cam = IS_cam[keys]
            IS_cam2 = IS_cam2[keys]
            seeds1 = seeds1[keys]
            seeds2 = seeds2[keys]

            IS_cam /= F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5
            IS_cam2 /= F.adaptive_max_pool2d(IS_cam2, (1, 1)) + 1e-5

            norm_cam = norm_cam.cpu().numpy()
            IS_cam = IS_cam.cpu().numpy()
            IS_cam2 = IS_cam2.cpu().numpy()
            seeds1 = seeds1.cpu().numpy()
            seeds2 = seeds2.cpu().numpy()

            draw_heatmap(norm_cam.copy(), keys, orig_img, f"{save_path}/cam", img_name)
            draw_heatmap(IS_cam.copy(), keys, orig_img, f"{save_path}/IScam", img_name)
            draw_heatmap(IS_cam2.copy(), keys, orig_img, f"{save_path}/refined_IScam", img_name)
            draw_seedregion(seeds1.copy(), orig_img, f"{save_path}/seeds1", img_name, keys)
            draw_seedregion(seeds2.copy(), orig_img, f"{save_path}/seeds2", img_name, keys)

            print(f"saved {pack['img_path'][0]}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_SIPE_refined2_1_hie_seed_norm_iscam2", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--ckpt", default="best.pth", type=str)
    parser.add_argument("--visualize", default=True, type=bool)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--make_dataset", default='train', type=str) # trainaug
    

    args = parser.parse_args()

    os.makedirs(os.path.join(args.session_name, 'npy'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'visual'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, 'infer.log'))
    print(vars(args))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '../PascalVOC2012/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'SEED_N_CAM')(num_cls=21)
        dataset = data_voc.VOC12ClsDatasetMSF(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, ))

    elif args.dataset == 'coco':
        dataset_root = '../coco2014/images'
        model = getattr(importlib.import_module(args.network), 'Feature')(num_cls=81)
        dataset = data_coco.COCOClsDatasetMSF('data/train_' + args.dataset + '.txt', coco_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))

    checkpoint = torch.load(args.session_name + '/ckpt/' + args.ckpt)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='') 
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()