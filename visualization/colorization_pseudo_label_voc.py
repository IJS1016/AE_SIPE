import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio

# session_path = 'SIPE_refined2_1_hie_seed_norm_iscam2'
# label_sesssion = 'pseudo_mask0.8_seg_label_0.8_71.00'

session_path = 'paper_reproduce'
label_sesssion = 'pseudo_label'

image_path = "/home/jsim/Desktop/datasets/voc2012_seg/trainaug_all_images/"
label_path = "/home/jsim/Desktop/datasets/voc2012_seg/trainaug_all_labels/"

VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                        (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)


def draw_segmap(seg_map, gt_label, orig_img, save_path, img_name):
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

if __name__ == "__main__" :
    pseudo_label_path = f"{session_path}/{label_sesssion}/"
    # pseudo_label_path = f"{session_path}/ir_label/"

    np_files = os.listdir(pseudo_label_path)
    np_files.sort()

    save_path = f"{session_path}/{label_sesssion}_color"
    os.makedirs(save_path, exist_ok=True)

    for iter, nf in enumerate(np_files) :
        if iter > 500 :
            exit()
        img_name = nf.split(".")[0]

        img_sem = np.array(Image.open(f"{pseudo_label_path}/{img_name}.png").convert('RGB'))
        orig_img = np.array(Image.open(f"{image_path}/{img_name}.jpg").convert('RGB'))
        orig_lbl = np.array(Image.open(f"{label_path}/{img_name}.png").convert('RGB'))


        img_sem = img_sem[:,:,0]
        img_sem = np.where(img_sem == 255, 21, img_sem)
        img_sem = VOC_color[img_sem]
        draw_segmap(img_sem, orig_lbl, orig_img, save_path, img_name)

        