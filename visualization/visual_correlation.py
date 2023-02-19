
import os
from builtins import bool
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

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

"""
Visualization CAM from SIPE model
have to set network, session name, etc...
"""

VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                        (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)


image_path = "/home/jsim/Desktop/WSSS/PascalVOC2012/VOCdevkit/VOC2012/JPEGImages/"

cudnn.enabled = True

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

def draw_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    for _, gt in enumerate(gt_label):
        heatmap = overlap(orig_img, norm_cam[gt])
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))
        imageio.imsave(cam_viz_path, heatmap)

def draw_segmap(seg_map, orig_img, save_path, img_name):
    seg_map = np.array(Image.fromarray((seg_map).astype(np.uint8), 'RGB').resize((orig_img.shape[1], orig_img.shape[0]), Image.BICUBIC))

    if seg_map.shape == orig_img.shape:
        out = (seg_map + np.array(orig_img).astype(np.float)) / 3
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
    save_path = 'Results_correlation'
    os.makedirs(os.path.join(save_path), exist_ok=True)

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        model.training = False

        for iter, pack in enumerate(data_loader):
            # use just one scale image ##################################
            # if iter > 50 :
            #     exit()
            if pack['name'][0] == "2007_000032" :
                img_name = pack['name'][0]
                label = pack['label'][0]
                size = pack['size']
                img = pack['img'][0].cuda()
                label = F.pad(label, (1, 0), 'constant', 1.0)
                keys = torch.nonzero(label).flatten() 
                label = label.cuda().unsqueeze(-1).unsqueeze(-1)                                          
                denorm_img =  denormalization(pack['img'][0]).cuda()
                
                h, w, correlation_map = model(img, label, denorm_img)      

                orig_img = np.array(Image.open(f"{image_path}/{img_name}.jpg").convert('RGB'))        

                correlation_map = correlation_map.view(h, w, h, w)

                for x in range(w) :
                    for y in range(h) :
                        target_corr = correlation_map[y, x]
                        target_corr /= target_corr.max()

                        target_corr = F.interpolate(target_corr.unsqueeze(0).unsqueeze(0), size, mode='bicubic', align_corners=False)
                        target_corr = target_corr.squeeze().cpu().numpy()

                        heatmap = overlap(orig_img, target_corr)
                        cam_viz_path = os.path.join(save_path, f'{img_name}_{x}_{y}.png')

                        heatmap[y*16:y*16+15,x*16:x*16+15] = [255, 255, 255]
                        imageio.imsave(cam_viz_path, heatmap)

                print(f"saved {img_name}")
                break



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_SIPE_refined2_1_norm_iscam2_miou_updated_cam_img_cls", type=str)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    # parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--ckpt", default="best.pth", type=str)
    parser.add_argument("--visualize", default=True, type=bool)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--make_dataset", default='train', type=str) # trainaug
    parser.add_argument("--single_img", default=True, type=bool) # trainaug

    args = parser.parse_args()

    pyutils.Logger(os.path.join(args.session_name, f'cam_infer__{args.make_dataset}.log'))
    print(vars(args))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '../PascalVOC2012/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'COORELATION')(num_cls=21)
        if args.single_img :
            dataset = data_voc.VOC12ClsDatasetMSF(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, ))
        else :
            dataset = data_voc.VOC12ClsDatasetMSF(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))

    elif args.dataset == 'coco':
        dataset_root = '../coco2014/images'
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=81)
        # dataset = data_coco.COCOClsDatasetMSF('data/train_' + args.dataset + '.txt', coco_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        dataset = data_coco.COCOClsDatasetMSF(f'data/{args.make_dataset}_' + args.dataset +'.txt', coco_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))

    checkpoint = torch.load(args.session_name + '/ckpt/' + args.ckpt)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='') 
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()