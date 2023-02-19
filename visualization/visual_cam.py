import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio

"""
Visualization CAM from cam numpy files
"""

session_path = 'SIPE_refined2_1_hie_seed_norm_iscam2'
image_path = "/home/jsim/Desktop/datasets/voc2012_seg/trainaug_all_images/"

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
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(keys[gt]))
        imageio.imsave(cam_viz_path, heatmap)

        print(f"finish {cam_viz_path}")

if __name__ == "__main__" :
    cam_np_path = f"{session_path}/npy_trainaug/"

    np_files = os.listdir(cam_np_path)
    np_files.sort()
    save_path = f"{session_path}/visual"
    os.makedirs(save_path, exist_ok=True)

    for iter, nf in enumerate(np_files) :
        if iter > 100 :
            exit()
        img_name = nf.split(".")[0]
        cam_dict = np.load(f"{cam_np_path}/{nf}", allow_pickle=True).item()
        cams = cam_dict['IS_CAM']
        keys = cam_dict['keys']

        orig_img = np.array(Image.open(f"{image_path}/{img_name}.jpg").convert('RGB'))

        draw_heatmap(cams, keys, orig_img, save_path, img_name)

        