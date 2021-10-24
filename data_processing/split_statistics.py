#split-statistics
import numpy as np
import pyexr

def create_item_statistics(path, mode='points'):
    if 'points' in mode:
        rgb = np.load(path+'/surface_points_blender.npz')['rgb'].astype(np.float32)
        means = np.mean(rgb, axis=0)
        stds = np.std(rgb,axis=0)
    if 'image' in mode:
        masked = False
        #load inputs per image, compute over all images
        rgb_all = None
        for i in range(30):
            rgb = pyexr.read(path+f'/_r_{str((i)*12).zfill(3)}.exr').astype(np.float32)
            #depthmap = pyexr.read(path+f'/_r_{str((i)*12).zfill(3)}_depth0001.exr').astype(np.float32)[...,0]
        
            #mask = rgb[...,3].copy().flatten()
            if masked:
                mask = rgb[...,3].copy().flatten()
                #mask = depthmap.flatten()
                rgb = rgb[mask<65000,:]
                rgb = rgb[mask==1,:]
            
            rgb = rgb[...,:3]/3            
            rgb = rgb.reshape(-1,3)

            if rgb_all is None: rgb_all = rgb
            else: rgb_all = np.concatenate((rgb_all, rgb), axis=0)
                
        means = np.mean(rgb_all, axis=0, keepdims=True)
        stds = np.std(rgb_all, axis=0, keepdims=True)

    return means, stds

"""def create_split_statistics(splitsfile, mode='points'):
    pass"""


if __name__ == '__main__':
    from util.arguments import argparse
    from tqdm import tqdm
    from joblib import Parallel, delayed
    parser = argparse.ArgumentParser(
        description="""Computes statistics (mean, std) for a splitsfile"""
    )
    #parser.add_argument('--inpath', type=str, default='../data/ShapeNetCore.v2')
    parser.add_argument('--splitsdir', type=str, default='../data/splits/chibane_shapenet')    
    args = parser.parse_args()
    splitsdir = args.splitsdir
    
    #https://cs231n.github.io/neural-networks-2/
    # Common pitfall. An important point to make about the preprocessing is that any preprocessing statistics 
    # (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data. 
    # E.g. computing the mean and subtracting it from every image across the entire dataset and then splitting the data 
    # into train/val/test splits would be a mistake. Instead, the mean must be computed only over the training data and 
    # then subtracted equally from all splits (train/val/test).
    splits = ['train']
    items = []
    for split in splits:
        with open(f'{splitsdir}/{split}.txt') as f:
            items += f.readlines()
    items = [line.replace('\n','') for line in items]
    means_points = np.zeros((1,3))
    stds_points = np.zeros((1,3))
    means_images = np.zeros((1,3))
    stds_images = np.zeros((1,3))
    
    n = len(items[:])
    for path in tqdm(items):
        mean_pts, std_pts = create_item_statistics(path)        
        means_points += mean_pts / n
        stds_points += std_pts / n

        mean_imgs, std_imgs = create_item_statistics(path, 'image')
        means_images += mean_imgs / n
        stds_images += std_imgs / n
    
    print(means_points, stds_points)
    print(means_images, stds_images)
    
    #mean, std = Parallel(n_jobs=8)(delayed(create_item_statistics)(i) for i in tqdm(names))
    """
    statistics: train split for blender_copy_fixed
    points [[0.36911592, 0.31920254, 0.30284569]] [[0.23451681, 0.21427469, 0.20948004]]
    images [[0.11082486, 0.09608462, 0.09117579]] [[0.2149361 , 0.19069198, 0.18358029]]
    """
    """
    Chibane
    points [[0.36580914 0.31664773 0.30158689]] [[0.2338024  0.21394474 0.20895862]]
    images [[0.10992143 0.09546989 0.09096114]] [[0.21359299 0.18972058 0.18306873]]
    """

    #print(textbody)

    
