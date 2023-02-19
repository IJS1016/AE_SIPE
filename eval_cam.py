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

def run(args, predict_dir, num_cls):
    preds = []
    masks = []
    n_images = 0
    for iter, pack in tqdm(enumerate(dataloader)):
        n_images += 1
        cam_dict = np.load(os.path.join(predict_dir, pack['name'][0] + '.npy'), allow_pickle=True).item()
        cams = cam_dict['IS_CAM'].squeeze() # squeeze added from JS
        keys = cam_dict['keys']
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

        mask = np.array(Image.open(os.path.join(args.gt_path,  pack['name'][0] + '.png')))
        masks.append(mask.copy())

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
    parser.add_argument('--ver', default="", type=str)
    parser.add_argument("--make_dataset", default='train', type=str) 
    args = parser.parse_args()

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '../PascalVOC2012/VOCdevkit/VOC2012'
        num_cls = 21
        dataset = data_voc.VOC12ImageDataset(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, img_normal=None, to_torch=False)
        # dataset = data_voc.VOC12ImageDataset('data/train_' + args.dataset + '.txt', voc12_root=dataset_root, img_normal=None, to_torch=False)

    elif args.dataset == 'coco':
        args.gt_path = "../coco2014/SegmentationClass/train2014/"
        dataset_root = "../coco2014/images"
        num_cls = 81
        dataset = data_coco.COCOImageDataset(f'data/{args.make_dataset}_' + args.dataset + '.txt', coco_root=dataset_root, img_normal=None, to_torch=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    # pyutils.Logger(os.path.join(args.session_name, 'eval_cam.log'))
    pyutils.Logger(os.path.join(args.session_name, f'eval_cam_{args.make_dataset}' + args.session_name +  f"{args.ver}" + '.log'))
    run(args, args.session_name + f'/npy_{args.make_dataset}' + f"{args.ver}", num_cls)
    # run(args, args.session_name + f'/npy_trainaug' + f"{args.ver}", num_cls)