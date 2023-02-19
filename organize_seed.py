import os
from PIL import Image
from IPython.display import Image as Img
from IPython.display import display
from PIL import Image, ImageFont, ImageDraw

iter_num = ['cam']
iter_num.extend([i for i in range(1,5)])
iter_num.append('iscam')
# base_dir = 'Results_miou_updated_cam_weight_SIPE_refined2_1_hie_seed_norm_iscam2_img_cls'
# base_dir = 'Results_similarity_updated_cam_weight_SIPE_refined2_1_hie_seed_norm_iscam2_img_cls'
# base_dir = 'Results_miou_bg_1_updated_cam_weight_SIPE_refined2_1_hie_seed_norm_iscam2_img_cls'

base_dir = 'Results_agglomerativeclustering/seed'
iter_num = [i for i in range(9)]


VOC_CAT_LIST = ['bg', 'aero', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train','tvmonitor']

fnt = ImageFont.truetype("/home/jsim/Downloads/arial.ttf", size=20)

save_dir = f"{base_dir}/results"
os.makedirs(save_dir, exist_ok=True)

tmp_dir = f"{base_dir}/{iter_num[0]}/"
# img_list = ["_".join(file_name[:-4].split("_")[:2]) for file_name in os.listdir(tmp_dir)]
file_list = os.listdir(tmp_dir)
file_list.sort()

img_list = []

for fn in file_list :
    img_name0, img_name1 = fn[:-4].split("_")
    img_name = f"{img_name0}_{img_name1}" 

    img_list.append(img_name)

for img_name in img_list :
    w_margin = 50
    h_margin = 30

    img_cam_path = f"{base_dir}/{iter_num[0]}/{img_name}.png"
    im0 = Image.open(img_cam_path)
    w, h = im0.width, im0.height

    dst = Image.new('RGB', (w_margin + w * len(iter_num), h_margin + h)) 
    draw = ImageDraw.Draw(dst)

    for x, th in enumerate(iter_num) :
        draw.text((int(w * (x + 0.5)), 0), str(th), font=fnt, fill="white")

        th_dir = f"{base_dir}/{th}/{img_name}.png"
                
        im0 = Image.open(th_dir)
        dst.paste(im0, (w_margin + x * w, h_margin))

    dst.save(f'{save_dir}/{img_name}.jpg', "JPEG")


