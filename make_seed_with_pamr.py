
import os
from builtins import bool
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import importlib

import imageio
import argparse
from data import data_voc, data_coco
from tool import torchutils, pyutils

cudnn.enabled = True

def overlap(img, seed_map):
    seed_map = seed_map.squeeze()
    seed_map = seed_map.astype(int)

    color = np.array([[0, 0, 0], [255, 0, 0]])
    seed_map = color[seed_map] * 2

    if seed_map.shape == np.array(img).astype(np.float).shape:
        out = (seed_map + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(seed_map.shape)
        print(np.array(img).shape)
    return out

def draw_seedmap(seed_map, gt_label, orig_img, save_path, img_name):
    
    seed_map = F.interpolate(seed_map, orig_img.shape[:-1], mode='bilinear', align_corners=False)
    seed_map = seed_map.squeeze()
    seed_map = seed_map.numpy()

    gt_cat = np.where(gt_label==1)[0]

    for _, gt in enumerate(gt_cat):
        heatmap = overlap(orig_img, seed_map[gt])
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))
        imageio.imsave(cam_viz_path, heatmap)

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
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    os.makedirs(os.path.join(args.session_name, "visual_per_size"), exist_ok=True)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        model.training = False

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            label = F.pad(label, (1, 0), 'constant', 1.0)

            # original
            #outputs = [model(img[0].cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)) for img in pack['img']]
            
            # print(pack['img'].shape) # torch.Size([1, 2, 3, 281, 500])
            
            # with PAMR
            label_cuda = label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
            img = pack['img'][0]
            outputs = model(img.cuda(non_blocking=True), label_cuda, denormalization(img).cuda(non_blocking=True))

            # multi-scale fusion
            seeds_x1 = outputs[0].cpu()
            seeds_x2 = outputs[1].cpu()
            seeds_x3 = outputs[2].cpu()
            seeds_x4 = outputs[3].cpu()
            seeds_hie = outputs[4].cpu()

            # visualize seed map
            if args.visualize:
                orig_img = np.array(Image.open(pack['img_path'][0]).convert('RGB'))
                draw_seedmap(seeds_x1, label, orig_img, os.path.join(args.session_name, 'visual'), f"{img_name}_x1")
                draw_seedmap(seeds_x2, label, orig_img, os.path.join(args.session_name, 'visual'), f"{img_name}_x2")
                draw_seedmap(seeds_x3, label, orig_img, os.path.join(args.session_name, 'visual'), f"{img_name}_x3")
                draw_seedmap(seeds_x4, label, orig_img, os.path.join(args.session_name, 'visual'), f"{img_name}_x4")
                draw_seedmap(seeds_hie, label, orig_img, os.path.join(args.session_name, 'visual'), f"{img_name}_hie")

            if process_id == n_gpus - 1 and iter % (len(databin) // 100) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 100)), end='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_AE_SIPE", type=str)
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--ckpt", default="final.pth", type=str)
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
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=21)
        # dataset = data_voc.VOC12ClsDatasetMSF(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        dataset = data_voc.VOC12ClsDatasetMSF(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, ))

    elif args.dataset == 'coco':
        dataset_root = "../ms_coco_14&15/images"
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=81)
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