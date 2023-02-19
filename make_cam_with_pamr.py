
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
    gt_cat = np.where(gt_label==1)[0]
    for _, gt in enumerate(gt_cat):
        heatmap = overlap(orig_img, norm_cam[gt])
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

            # except already saved
            if os.path.isfile(os.path.join(args.session_name, f'npy_{args.make_dataset}', img_name + '.npy')) :
                try : 
                    np.load(os.path.join(os.path.join(args.session_name, f'npy_{args.make_dataset}', img_name + '.npy')), allow_pickle=True).item()
                    if process_id == n_gpus - 1 and iter % (len(databin) // 100) == 0:
                        print("%d " % ((5*iter+1)//(len(databin) // 100)), end='')
                    continue
                except :
                    pass

            label = pack['label'][0]
            size = pack['size']
            label = F.pad(label, (1, 0), 'constant', 1.0)

            # original
            #outputs = [model(img[0].cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)) for img in pack['img']]
            
            # with PAMR
            label_cuda = label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
            outputs = [model(img[0].cuda(non_blocking=True), label_cuda, denormalization(img[0]).cuda(non_blocking=True)) for img in pack['img']]

            # multi-scale fusion
            IS_CAM_list = [output[1].cpu() for output in outputs] # [cls, h ,w]
            IS_CAM_list = [F.interpolate(torch.unsqueeze(o, 1), size, mode='bilinear', align_corners=False) for o in IS_CAM_list]

            # JS check diff scale image cam ################################################################################################################
            # visualization each multi scale
            # save_image_num = 20
            # scale = (1.0, 0.5, 1.5, 2.0)

            # orig_img = np.array(Image.open(pack['img_path'][0]).convert('RGB'))

            # if iter > save_image_num :
            #     return None

            # for aug_idx, is_cam in enumerate(IS_CAM_list) : 
            #     is_cam /= F.adaptive_max_pool2d(is_cam, (1, 1)) + 1e-5
            #     is_cam = is_cam.cpu().numpy()
            #     draw_heatmap(is_cam.copy(), label, orig_img, os.path.join(args.session_name, "visual_per_size"), f"{img_name}_S{scale[aug_idx]}")
            ################################################################################################################################################
            
            IS_CAM = torch.sum(torch.stack(IS_CAM_list, 0), 0)[:,0]
            IS_CAM /= F.adaptive_max_pool2d(IS_CAM, (1, 1)) + 1e-5
            IS_CAM = IS_CAM.cpu().numpy()

            # visualize IS-CAM
            # if args.visualize:
            #     orig_img = np.array(Image.open(pack['img_path'][0]).convert('RGB'))
            #     draw_heatmap(IS_CAM.copy(), label, orig_img, os.path.join(args.session_name, 'visual'), img_name)

            # save IS_CAM
            valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
            IS_CAM = IS_CAM[valid_cat]
            np.save(os.path.join(args.session_name, f'npy_{args.make_dataset}', img_name + '.npy'),  {"keys": valid_cat, "IS_CAM": IS_CAM})
            #print(iter)

            if process_id == n_gpus - 1 and iter % (len(databin) // 100) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 100)), end='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_AE_SIPE", type=str)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    # parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--ckpt", default="best.pth", type=str)
    parser.add_argument("--visualize", default=True, type=bool)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--make_dataset", default='train', type=str) # trainaug

    args = parser.parse_args()

    os.makedirs(os.path.join(args.session_name, f'npy_{args.make_dataset}'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'visual'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, f'cam_infer__{args.make_dataset}.log'))
    print(vars(args))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '../PascalVOC2012/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=21)
        dataset = data_voc.VOC12ClsDatasetMSF(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        # dataset = data_voc.VOC12ClsDatasetMSF(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, ))

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