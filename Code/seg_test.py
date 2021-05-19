import glob
import os
import torch
import nibabel as nib
import ipdb
from seg.unet import Unet, AttU_Net
from utils.core import norm, wct_style_transfer
import numpy as np
from torchvision import transforms
import cv2
from seg.seg_utils import remove_minor_cc
from utils.wavelet_model import WaveEncoder, WaveDecoder
import ttach as tta
import time
# CUDA_VISIBLE_DEVICES=0 python seg_test.py
model = Unet().cuda()
model_path = 'fintune_pth_model/model_50.pth'
print(model_path)
model.load_state_dict(torch.load(model_path))
model.eval()
tta_transforms = tta.Compose(
    [
            tta.Rotate90(angles=[90, 180, 270])
        ]
)
tta_model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean')
num = 1
for v in ['D', 'C', 'B', 'A']:
    ven = f'ven_{v}'
    print(ven)
    ids_all = glob.glob(os.path.join(f'/data/hxq/zd_code_data/data/mnms/Test/nii_data_ven/{ven}/image', '*_sa_??.nii.gz'))
    for ids in ids_all:
        ids_name = os.path.basename(ids)
        vol = nib.load(ids)
        vol_fdata, vol_aff, vol_head = vol.get_fdata(), vol.affine, vol.header
        h, w = vol_fdata.shape[:2]
        vol_fdata = cv2.resize(vol_fdata, (256, 256))
        vol_data = torch.from_numpy(vol_fdata).float() # [196, 240, 12]
        
        pred_all = []
        for idx in range(vol_data.shape[-1]):
            image = vol_data[:, :, idx].unsqueeze(0).unsqueeze(0) # [196, 240]
            image = (image-torch.min(image))/(torch.max(image)-torch.min(image))        

            img_norm = norm(image.cuda())
            s_time = time.time()
            output = tta_model(img_norm)
            e_time = time.time()
            print(e_time-s_time)
            ipdb.set_trace()
            pred = torch.argmax(output, 1).data.cpu().numpy().squeeze(0)
            # ipdb.set_trace()
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_all.append(pred)
        pred_all = np.stack(pred_all, axis=0).transpose(1, 2, 0)
        pred_all = remove_minor_cc(pred_all)
        pred_nii = nib.Nifti1Image(pred_all*85, affine=vol_aff, header=vol_head)
        save_dir = f'pred_output/{ven}/image{str(num)}/'
        os.makedirs(save_dir, exist_ok=True)
        if v == 'C':
            # print('c_dir')
            save_dir = f'pred_output/{ven}/image{str(num+1)}/'
            os.makedirs(save_dir, exist_ok=True)        
        pred_nii.to_filename(save_dir+ids_name)



