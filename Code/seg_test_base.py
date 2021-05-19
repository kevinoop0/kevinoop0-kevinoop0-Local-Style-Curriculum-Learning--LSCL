import glob
import os
import torch
import nibabel as nib
import ipdb
from seg.unet import Unet, AttU_Net
from utils.core import norm
import numpy as np
from torchvision import transforms
import cv2
# CUDA_VISIBLE_DEVICES=2 python seg_test_base.py
model = Unet().cuda()
model.load_state_dict(torch.load('model_checkpoints/unet/model_2.pth'))
model.eval()

for v in ['A', 'B', 'C', 'D']:
    ven = f'ven_{v}'
    print(ven)
    ids_all = glob.glob(os.path.join(f'/data/hxq/zd_code_data/data/mnms/Test/nii_data_ven/{ven}/image', '*_sa_??.nii.gz'))
    # print(ids_all)
    for ids in ids_all:
        ids_name = os.path.basename(ids)
        vol = nib.load(ids)
        vol_fdata, vol_aff, vol_head = vol.get_fdata(), vol.affine, vol.header
        vol_data = torch.from_numpy(vol_fdata).float() # [196, 240, 12]
        h, w = vol_data.shape[:2]
        pred_all = []
        for idx in range(vol_data.shape[-1]):
            image = vol_data[:, :, idx].unsqueeze(0).unsqueeze(0) # [196, 240]
            image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
            # image = torch.cat([image, image, image], dim=1)
            image = transforms.Resize((256, 256))(image.cuda())
            # cv2.imwrite('img.png', np.uint8((image[0]*255.0).cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)))
            img_norm = norm(image)           
            output = model(img_norm)
            pred = torch.argmax(output, 1).data.cpu().numpy().squeeze(0)
            # ipdb.set_trace()
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_all.append(pred)
        pred_all = np.stack(pred_all, axis=0).transpose(1, 2, 0)
        # pred_all = remove_minor_cc(pred_all)
        pred_nii = nib.Nifti1Image(pred_all*85, affine=vol_aff, header=vol_head)
        pred_nii.to_filename(f'pred_output/{ven}/image/{ids_name}')
