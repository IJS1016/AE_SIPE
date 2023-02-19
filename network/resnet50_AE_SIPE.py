import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50
import random
from network.pamr import PAMR

class Net(nn.Module):

    def __init__(self, num_cls=21):
        super(Net, self).__init__()

        self.num_cls = num_cls

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)
        self.classifier = nn.Conv2d(2048, self.num_cls-1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.side1, self.side2, self.side3, self.side4])

        self._aff = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda() # same as PAMR paper setting
        # https://github.com/visinf/1-stage-wseg/blob/cfe5784f9905d656e0f15fba0e6eb76a3731d80f/models/mods/pamr.py


    def get_seed(self, norm_cam, label, feature):
        
        n,c,h,w = norm_cam.shape

        # iou evalution
        seeds = torch.zeros((n,h,w,c)).cuda()
        feature_s = feature.view(n,-1,h*w)
        feature_s = feature_s/(torch.norm(feature_s,dim=1,keepdim=True)+1e-5)
        correlation = F.relu(torch.matmul(feature_s.transpose(2,1), feature_s),inplace=True).unsqueeze(1) #[n,1,h*w,h*w]
        # correlation = correlation/torch.max(correlation, dim=-1)[0].unsqueeze(-1) #[n,1,h*w,h*w]
        cam_flatten = norm_cam.view(n,-1,h*w).unsqueeze(2) #[n,21,1,h*w]
        inter = (correlation * cam_flatten).sum(-1)
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter/union).view(n,self.num_cls,h,w) #[n,21,h,w]
        miou[:,0] = miou[:,0]*0.5
        probs = F.softmax(miou, dim=1)
        belonging = miou.argmax(1)
        seeds = seeds.scatter_(-1, belonging.view(n,h,w,1), 1).permute(0,3,1,2).contiguous()
        
        seeds = seeds * label
        return seeds, probs
    

    # FOR COCO####################################################################################################################
    # def get_seed(self, norm_cam, label, feature):
        
    #     n,c,h,w = norm_cam.shape

    #     # iou evalution
    #     feature_s = feature.view(n,-1,h*w)
    #     feature_s = feature_s/(torch.norm(feature_s,dim=1,keepdim=True)+1e-5)
    #     correlation = F.relu(torch.matmul(feature_s.transpose(2,1), feature_s),inplace=True).unsqueeze(1) #[n,1,h*w,h*w]

    #     del feature
    #     del feature_s

    #     # correlation = correlation/torch.max(correlation, dim=-1)[0].unsqueeze(-1) #[n,1,h*w,h*w]
    #     cam_flatten = norm_cam.view(n,-1,h*w).unsqueeze(2) #[n,21,1,h*w]

    #     # to prevent out of memory with coco dataset
    #     # if h * w > 3000 :
    #     #     inter = torch.zeros(n, c, h*w).to(cam_flatten.device)
    #     #     for _c in range(c) :
    #     #         inter[:,_c] = (correlation * cam_flatten[:,_c,None]).sum(-1)
    #     # else :
    #     #     inter = (correlation * cam_flatten).sum(-1)
        
    #     inter = torch.zeros(n, c, h*w).to(cam_flatten.device)
    #     for _c in range(c) :
    #         inter[:,_c] = (correlation * cam_flatten[:,_c,None]).sum(-1)
        
    #     # for sanity checking
    #     # _inter = torch.zeros(n, c, h*w).to(correlation.device)
    #     # for _c in range(c) :
    #     #     _inter[:,_c] = (correlation * cam_flatten[:,_c,None]).sum(-1)
    #     # inter = (correlation * cam_flatten).sum(-1)

    #     # inter = (correlation * cam_flatten).sum(-1)
    #     union = correlation.sum(-1) + cam_flatten.sum(-1) - inter

        
    #     miou = (inter/union).view(n,self.num_cls,h,w) #[n,21,h,w]
    #     miou[:,0] = miou[:,0]*0.5 
    #     probs = F.softmax(miou, dim=1)
    #     belonging = miou.argmax(1)

    #     seeds = torch.zeros((n,h,w,c)).cuda()
    #     seeds = seeds.scatter_(-1, belonging.view(n,h,w,1), 1).permute(0,3,1,2).contiguous()
        
    #     seeds = seeds * label
    #     return seeds, probs
    ####################################################################################################################
    
    def get_prototype(self, seeds, feature):
        ch = 768
        n,c,h,w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1)  # seed:[n,21,1,h,w], feature:[n,1,c,h,w], crop_feature:[n,21,c,h,w]
                                                                  # crop_feaute shape with VOC torch.Size([2, 21, 768, 32, 32])
        # Original Code
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1,c,h,w), (1,1)).view(n, self.num_cls, c, 1, 1) # prototypes:[n,21,c,1,1]
        return prototype


    def reactivate(self, prototype, feature):
        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2)) # feature:[n,1,c,h,w], prototypes:[n,21,c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam

    def run_pamr(self, im, mask): 
        N, C, H, W = im.shape
        im = F.interpolate(im, (H // 4, W // 4), mode="bilinear", align_corners=True)
        masks_dec = self._aff(im, mask)

        return masks_dec


    def forward(self, x, valid_mask, x_raw=None, labels=None):
        test_mode = labels is None

        # valid_mask [16. 21. 32. 32]
        N, C, H, W = x.size()

        # forward
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())        
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        # sem_feature = x4
        sem_feature = hie_fea
        cam = self.classifier(x4)
        score = F.adaptive_avg_pool2d(cam, 1)

        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)*valid_mask

        seeds, probs = self.get_seed(norm_cam.clone(), valid_mask.clone(), sem_feature.clone())
        # seed : torch.Size([16, 21, 32, 32])
        # hie_fea : torch.Size([16, 768, 32, 32])
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)

        masks_dec = self.run_pamr(x_raw, IS_cam.detach())

        # IS_cam 2
        masks_dec = F.interpolate(masks_dec, side3.shape[2:], mode='bilinear', align_corners=True)
        dec_argmax_mask = masks_dec.argmax(1)
        dec_bin_mask = torch.zeros_like(seeds)
        
        for c in range(self.num_cls) :
            dec_bin_mask[:, c, :, :] = (dec_argmax_mask == c)

        prototypes2 = self.get_prototype(dec_bin_mask, hie_fea)
        IS_cam2 = self.reactivate(prototypes2, hie_fea)

        IS_cam2 = IS_cam2/(F.adaptive_max_pool2d(IS_cam2, (1, 1)) + 1e-5)

        # IS_cam 2
        # IS_argmax_mask = IS_cam.argmax(1)
        # IS_bin_mask = torch.zeros_like(seeds)
    
        # for c in range(self.num_cls) :
        #     IS_bin_mask[:, c, :, :] = (IS_argmax_mask == c)

        return {"score": score, "cam" : norm_cam, "seeds" : seeds, "prototypes": prototypes2, "IS_cam" : IS_cam2, "probs" : probs, "pamr" : masks_dec} #, "IS_cam_bf_PAMR" : IS_cam}
    
    def _rescale_and_clean(self, masks, image, labels):
        """Rescale to fit the image size and remove any masks
        of labels that are not present"""
        masks = F.interpolate(masks, size=image.size()[2:], mode='bilinear', align_corners=True)
        masks[:, 1:] *= labels.type_as(masks)
        return masks

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


# validation network
class CAM(Net):

    def __init__(self, num_cls):
        super(CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    def forward(self, x, label, x_raw=None):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())        
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        cam = self.classifier(x4)
        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)
        
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

        seeds, _ = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), hie_fea.clone())
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)

        masks_dec = self.run_pamr(x_raw[0].unsqueeze(0), IS_cam.detach())

        # IS_cam 2
        masks_dec = F.interpolate(masks_dec, side3.shape[2:], mode='bilinear', align_corners=True)
        dec_argmax_mask = masks_dec.argmax(1)
        dec_bin_mask = torch.zeros_like(seeds)
        
        for c in range(self.num_cls) :
            dec_bin_mask[:, c, :, :] = (dec_argmax_mask == c)

        prototypes2 = self.get_prototype(dec_bin_mask, hie_fea)
        IS_cam2 = self.reactivate(prototypes2, hie_fea)

        IS_cam2 = IS_cam2/(F.adaptive_max_pool2d(IS_cam2, (1, 1)) + 1e-5)

        # return norm_cam[0], IS_cam[0]
        return norm_cam[0], IS_cam2[0], prototypes2[0].squeeze()

# validation network
class VARCAM(Net): # CAMs for different classification ratios

    def __init__(self, num_cls):
        super(VARCAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    def forward(self, x, label, x_raw=None):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())        
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        cam = self.classifier(x4)
        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)
        
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

        seeds, _ = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), hie_fea.clone())
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)

        masks_dec = self.run_pamr(x_raw[0].unsqueeze(0), IS_cam.detach())

        # IS_cam 2
        masks_dec = F.interpolate(masks_dec, side3.shape[2:], mode='bilinear', align_corners=True)
        dec_argmax_mask = masks_dec.argmax(1)
        dec_bin_mask = torch.zeros_like(seeds)
        
        for c in range(self.num_cls) :
            dec_bin_mask[:, c, :, :] = (dec_argmax_mask == c)

        prototypes2 = self.get_prototype(dec_bin_mask, hie_fea)
        IS_cam2 = self.reactivate(prototypes2, hie_fea)

        # IS_cam2 = IS_cam2/(F.adaptive_max_pool2d(IS_cam2, (1, 1)) + 1e-5)

        # return norm_cam[0], IS_cam[0]
        return norm_cam[0], IS_cam2[0], prototypes2[0].squeeze(), hie_fea


# for visualization seed map
class Feature(Net):

    def __init__(self, num_cls):
        super(Feature, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    def forward(self, x, label):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())        
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        # sem_feature = x4
        sem_feature = hie_fea
        cam = self.classifier(x4)
        
        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

        seeds_hie, probs = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), sem_feature.clone())
        seeds_last, probs = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), sem_feature.clone())

        return seeds_hie, seeds_last

    # # maybe correct?
    # def forward(self, x, label):

    #     x0 = self.stage0(x)
    #     x1 = self.stage1(x0)
    #     x2 = self.stage2(x1).detach()
    #     x3 = self.stage3(x2)
    #     x4 = self.stage4(x3)

    #     side1 = self.side1(x1.detach())
    #     side2 = self.side2(x2.detach())        
    #     side3 = self.side3(x3.detach())
    #     side4 = self.side4(x4.detach())

    #     hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
    #                           F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
    #                           F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
    #                           F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

    #     sem_feature = x4
    #     # sem_feature = hie_fea
    #     cam = self.classifier(hie_fea)
        
    #     # initialize background map
    #     norm_cam = F.relu(cam)
    #     norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
    #     cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
    #     norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
    #     norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

    #     # original?
    #     # seeds_hie, probs = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), sem_feature.clone())
    #     # seeds_last, probs = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), sem_feature.clone())

    #     seeds_hie, probs = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), hie_fea.clone())
    #     seeds_last, probs = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), sem_feature.clone())

    #     return seeds_hie, seeds_last



# for visualization seed map
class SEED_N_CAM(Net):
    def __init__(self, num_cls):
        super(SEED_N_CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    def forward(self, x, label, x_raw=None):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())        
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        cam = self.classifier(x4)
        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)
        
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

        seeds, _ = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), hie_fea.clone())
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)

        masks_dec = self.run_pamr(x_raw[0].unsqueeze(0), IS_cam.detach())

        # IS_cam 2
        masks_dec = F.interpolate(masks_dec, side3.shape[2:], mode='bilinear', align_corners=True)
        dec_argmax_mask = masks_dec.argmax(1)
        dec_bin_mask = torch.zeros_like(seeds)
        
        for c in range(self.num_cls) :
            dec_bin_mask[:, c, :, :] = (dec_argmax_mask == c)

        prototypes2 = self.get_prototype(dec_bin_mask, hie_fea)
        IS_cam2 = self.reactivate(prototypes2, hie_fea)

        return norm_cam, IS_cam, IS_cam2, seeds, dec_bin_mask # CAM, FIRST IS_CAM, SECOND IS_CAM, FIRST SEED MAP, SECEND SEED MAP
