import torch
import numpy as np
import random
import os
import pandas as pd
from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet101_2 
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, de_resnet101, de_resnet152, de_resnext50_32x4d, de_resnext101_32x8d, de_wide_resnet101_2
from utils.utils_test import evaluation_multi_proj, evaluation_sp
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import MVTecDataset_test, get_data_transforms
from datetime import datetime
np.bool = np.bool_

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default = './your_checkpoint_folder', type=str)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--classes', nargs="+", default=["carpet", "leather"])
    parser.add_argument('--model_type', default=['wide_resnet50_2'])
    pars = parser.parse_args()
    return pars

def inference(_class_, pars):
    model_type = pars.model_type
    if not os.path.exists(pars.checkpoint_folder):
        os.makedirs(pars.checkpoint_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    test_path = './content/' + _class_

    
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # set model
    encoder, bn = None, None
    decoder = None
    mpl_base = None

    if model_type in ['wide_resnet50_2']:
      encoder, bn = wide_resnet50_2(pretrained=True)
      decoder = de_wide_resnet50_2(pretrained=False)
      mpl_base = 64

    if model_type in ['resnet18']:
      encoder, bn = resnet18(pretrained=True)
      decoder = de_resnet18(pretrained=False)
      mpl_base = 16

    if model_type in ['resnet34']:
      encoder, bn = resnet34(pretrained=True)
      decoder = de_resnet34(pretrained=False)
      mpl_base = 16

    if model_type in ['resnet50']:
      encoder, bn = resnet50(pretrained=True)
      decoder = de_resnet50(pretrained=False)
      mpl_base = 64

    if model_type in ['wide_resnet101_2']:
      encoder, bn = wide_resnet101_2(pretrained=True)
      decoder = de_wide_resnet101_2(pretrained=False)
      mpl_base = 64

    # Use pretrained model for encoder

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    
    proj_layer =  MultiProjectionLayer(base=mpl_base).to(device)
    # Load trained weights for projection layer, bn (OCBE), decoder (student)    
    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + model_type+'_'+_class_+'.pth'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])

    start=datetime.now()
    #auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device) 
    auroc_px, auroc_sp, aupro_px = evaluation_sp(encoder, proj_layer, bn, decoder, test_dataloader, device)       
    print( datetime.now()-start)
    print('{}: Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(_class_, auroc_sp, auroc_px, aupro_px))
    return auroc_sp, auroc_px, aupro_px


if __name__ == '__main__':
    pars = get_args()

    item_list = [ 'carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
    setup_seed(111)
    metrics = {'Class': [], 'Model': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = inference(c, pars)
        metrics['Class'].append(c)
        metrics['Model'].append(pars.model_type)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
            
    df_metrics = pd.DataFrame(metrics)
    metrics_path = f'{pars.checkpoint_folder}/metrics_checkpoints.csv'
    if os.path.exists(metrics_path):
      df = pd.read_csv(metrics_path)
      df_metrics = pd.concat([df, df_metrics], ignore_index=True, sort=False)
    df_metrics.to_csv(metrics_path, index=False)