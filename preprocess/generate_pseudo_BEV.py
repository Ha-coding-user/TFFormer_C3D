import torch
import numpy as np
import pickle
import os
import argparse
import glob
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        # required=True,
        default='/nas1/hamyo/Data/kitti/dataset/labels/',
        help='dataset folder containing xxxxxx_1_2.npy'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        # required=True,
        default='/nas1/hamyo/Data/kitti/dataset/pseudo_bev/',
        help='output folder for generated sequence scans.'
    )
    
    parser.add_argument(
        '--is_save',
        type=bool,
        default=False,
        help='visualization for checking'
    )
    
    parser.add_argument(
        '--vis_path',
        '-v',
        type=str,
        default='/nas1/hamyo/Data/kitti/dataset/vis_bev/'
    )
    
    args = parser.parse_args()
    dataset = args.dataset
    output = args.output
    is_save = args.is_save
    vis_path = args.vis_path
    
    if is_save:
        sel_col = ['#000000','#6496f5','#64e6f5','#1e3c96','#501eb4','#0000ff','#ff1e1e','#ff28c9','#961e5a','#ff00ff','#ff96ff','#4b004b','#af004c','#ffc800','#ff7832','#00af00','#873c00', '#96f050',
                   '#ffef96', '#fff096']
        sel_colmap = ListedColormap(sel_col)
        sel_norm = list(range(0, 20, 1))
        sel_norm = BoundaryNorm(sel_norm, ncolors=len(sel_col))
        
    
    print("*" * 80)
    print("dataset folder\t:", args.dataset)
    print("output folder\t:", args.output)
    print("is_save\t:", args.is_save)
    if is_save:
        print("vis_path\t:", args.vis_path)
    print("*" * 80)
    
    exception = ['shadow', 'curvy', 'light', 'occlusion', 't']
    
    for sequence in sorted(os.listdir(dataset)):
        if sequence in exception:
            continue
        for data in tqdm(sorted(os.listdir(f'{dataset}/{sequence}/'))):
            id, resolution = data.split('_', 1)
            if resolution == '1_1.npy':
                continue
            
            label = np.load(f'{dataset}/{sequence}/{data}')
            bev_grid = np.zeros((128, 128))
            flag = np.zeros((128, 128))
            
            for height in range(label.shape[-1])[::-1]:
                cur_height = label[:, :, height]
                cur_flag = np.where((flag == 0) & (cur_height != 0) & (cur_height != 255), 1, 0)
                bev_grid = np.where((cur_flag == 1), cur_height, bev_grid)
                flag = np.where((flag == 1) | (cur_flag == 1), 1, 0)
            
            if is_save:
                if not os.path.exists(f'{vis_path}/{sequence}'):
                    os.makedirs(f'{vis_path}/{sequence}')
                
                plt.clf()
                sns.heatmap(bev_grid,
                            cmap=sel_colmap,
                            norm=sel_norm,
                            cbar=True)
                plt.savefig(f'{vis_path}/{sequence}/{id}.png')
                plt.clf()
                
            if not os.path.exists(os.path.join(output, sequence)):
                os.makedirs(os.path.join(output, sequence))
                
            np.save(os.path.join(output, sequence, id + '.npy'), bev_grid)
            print(f'wrote to {output}/{sequence}/{id}.npy')