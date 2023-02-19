import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from tool.imutils import crf_inference_label

"""
Visualization pseudo label map from cam after crf from cam numpy files
"""

session_path = 'SIPE_refined2_1_with_infer100'
image_path = "/home/jsim/Desktop/datasets/voc2012_seg/trainaug_all_images/"

VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                        (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)


def draw_segmap(seg_map, gt_label, orig_img, save_path, img_name):
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

if __name__ == "__main__" :
    cam_np_path = f"{session_path}/npy/"

    np_files = os.listdir(cam_np_path)
    np_files.sort()

    save_path = f"{session_path}/visual_crf_color"
    os.makedirs(save_path, exist_ok=True)

    for iter, nf in enumerate(np_files) :
        if iter > 100 :
            exit()
        img_name = nf.split(".")[0]
        cam_dict = np.load(f"{cam_np_path}/{nf}", allow_pickle=True).item()
        cams = cam_dict['IS_CAM']
        keys = cam_dict['keys']
        cls_labels = np.argmax(cams, axis=0)

        orig_img = np.array(Image.open(f"{image_path}/{img_name}.jpg").convert('RGB'))

        pred_crf = crf_inference_label(orig_img, cls_labels, n_labels=keys.shape[0])
        pred_crf = keys[pred_crf]

        pred_crf = VOC_color[pred_crf]
        draw_segmap(pred_crf, keys, orig_img, save_path, img_name)

        