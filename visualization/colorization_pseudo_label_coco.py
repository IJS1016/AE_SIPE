import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import matplotlib # For Matlab's color maps

# session_path = 'SIPE_refined2_1_hie_seed_norm_iscam2'
# label_sesssion = 'pseudo_mask0.8_seg_label_0.8_71.00'

session_path = 'SIPE_refined2_1_hie_seed_norm_iscam2_COCO'
label_sesssion = 'ir_label_0.8'

image_path = "/home/jsim/Desktop/WSSS/coco2014/images/train2014"
label_path = "SIPE_refined2_1_hie_seed_norm_iscam2_COCO/ir_label_0.8"

VOC_color = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                        (0, 192, 0), (128, 192, 0), (0, 64, 128), (255, 255, 255)], np.float32)
                        
def getCMap(stuffStartId=1, stuffEndId=81, cmapName='jet', addThings=True, addUnlabeled=True, addOther=True):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param stuffStartId: (optional) index where stuff classes start
    :param stuffEndId: (optional) index where stuff classes end
    :param cmapName: (optional) Matlab's name of the color map
    :param addThings: (optional) whether to add a color for the 91 thing classes
    :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
    :param addOther: (optional) whether to add a color for the 'other' class
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = stuffEndId - stuffStartId + 1
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black (or any other) color for each thing class
    if addThings:
        thingsPadding = np.zeros((stuffStartId - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

    # Add black color for 'unlabeled' class
    if addUnlabeled:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    # Add yellow/orange color for 'other' class
    if addOther:
        cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

    return cmap

def draw_segmap(seg_map, orig_img, save_path, img_name):
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
    color_map = getCMap()
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
        

        img_sem = img_sem[:,:,0]
        img_sem = np.where(img_sem == 255, 81, img_sem)
        # img_sem = VOC_color[img_sem]
        img_sem_colored = color_map[img_sem]
        
        draw_segmap(img_sem_colored, orig_img, save_path, img_name)

        