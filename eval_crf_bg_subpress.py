import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
from PIL import Image
from tool import pyutils
from data import data_voc, data_coco
from tqdm import tqdm

from tool.imutils import crf_inference_label
import imageio
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion

def run(args, predict_dir, num_cls, dataset_root):
    preds = []
    masks = []
    n_images = 0
    for iter, pack in tqdm(enumerate(dataloader)):
        n_images += 1
        cam_dict = np.load(os.path.join(predict_dir, pack['name'][0] + '.npy'), allow_pickle=True).item()
        cams = cam_dict['IS_CAM'].squeeze() # squeeze added from JS
        keys = cam_dict['keys']
        
        # for bg subpress
        cams[0] = cams[0] * args.subpress
        cls_labels = np.argmax(cams, axis=0)

        img_orig255 = np.asarray(imageio.imread(dataset_root + '/JPEGImages/' + pack['name'][0] + '.jpg'))
        # img_orig255 h, w, c (img, bg_conf_cam, n_labels=keys.shape[0])
        pred_crf = crf_inference_label(img_orig255, cls_labels, n_labels=keys.shape[0])
        
        pred_crf = keys[pred_crf]
        preds.append(pred_crf.copy())

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
    parser.add_argument('--subpress', default=0.8, type=float) # pascalVOC best setting 0.8
    parser.add_argument("--make_dataset", default='train', type=str) 
    args = parser.parse_args()

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '../PascalVOC2012/VOCdevkit/VOC2012'
        num_cls = 21
        dataset = data_voc.VOC12ImageDataset(f'data/{args.make_dataset}_' + args.dataset + '.txt', voc12_root=dataset_root, img_normal=None, to_torch=False)
        
    elif args.dataset == 'coco':
        args.gt_path = "../ms_coco_14&15/SegmentationClass/train2014/"
        dataset_root = '../coco2014/images'
        num_cls = 81
        dataset = data_coco.COCOImageDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root, img_normal=None, to_torch=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    pyutils.Logger(os.path.join(args.session_name, f'eval_with_crf_{args.make_dataset}' + args.session_name +  f"subpress_{args.subpress}" + '.log'))
    # run(args, args.session_name + f'/npy_{args.make_dataset}' + f"{args.ver}", num_cls, dataset_root)

    if os.path.isdir(args.session_name + f'/npy_trainaug') :
        run(args, args.session_name + f'/npy_trainaug', num_cls, dataset_root)
    elif os.path.isdir(args.session_name + f'/npy_{args.make_dataset}') :
        run(args, args.session_name + f'/npy_{args.make_dataset}', num_cls, dataset_root)
    else :
        run(args, args.session_name + f'/npy', num_cls, dataset_root)