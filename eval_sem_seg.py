
import numpy as np
import os
from tool import pyutils
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
import argparse

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)

    sem_seg_out_dir = args.session_name + "/" + args.sem_seg_dir

    preds = []
    labels = []
    n_img = 0
    
    for i, id in enumerate(dataset.ids):
        cls_labels = imageio.imread(os.path.join(sem_seg_out_dir, id + '.png')).astype(np.uint8)
        preds.append(cls_labels.copy())
        labels.append(dataset.get_example_by_keys(i, (1,))[0])
        n_img += 1

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print("total images", n_img)
    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))
    print({'iou': iou, 'miou': np.nanmean(iou)})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--voc12_root", default='/home/jsim/Desktop/datasets/VOCdevkit/VOC2012', type=str)
    parser.add_argument("--gt_dir", default='../PascalVOC2012/VOCdevkit/VOC2012/SegmentationClass', type=str)
    parser.add_argument("--chainer_eval_set", default="train", type=str)
    parser.add_argument('--session_name', default="exp", type=str)
    parser.add_argument("--sem_seg_dir", default="pseudo_mask0.8_seg_label_0.8", type=str)
    args = parser.parse_args()

    run(args)

    # pyutils.Logger(os.path.join(args.session_name, 'eval_' + args.session_name +  f"{args.sem_seg_dir}" + '.l og'))
    pyutils.Logger(os.path.join(args.session_name, 'eval_' + f"{args.sem_seg_dir}" + '.log'))