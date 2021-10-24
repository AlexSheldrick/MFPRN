import numpy as np
import trimesh
from pykdtree.kdtree import KDTree
from data_processing.implicit_waterproofing import implicit_waterproofing
from util.visualize import render_mesh
import pyexr
from skimage.color import rgba2rgb
rng = np.random.default_rng()

# taken from ifnet: https://github.com/jchibane/if-net/blob/master/data_processing/evaluation.py
## mostly apdopted from occupancy_networks/im2mesh/common.py and occupancy_networks/im2mesh/eval.py

def eval_mesh( mesh_pred, mesh_gt, bb_min, bb_max, n_points=100000):

    pointcloud_pred, idx = mesh_pred.sample(n_points, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_gt, idx = mesh_gt.sample(n_points, return_index=True)
    pointcloud_gt = pointcloud_gt.astype(np.float32)
    normals_gt = mesh_gt.face_normals[idx]

    out_dict = eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)


    bb_len = bb_max - bb_min
    bb_samples = np.random.rand(n_points*10, 3) * bb_len + bb_min

    occ_pred = implicit_waterproofing(mesh_pred, bb_samples)[0]
    occ_gt = implicit_waterproofing(mesh_gt, bb_samples)[0]

    area_union = (occ_pred | occ_gt).astype(np.float32).sum()
    area_intersect = (occ_pred & occ_gt).astype(np.float32).sum()

    out_dict['iou'] =  (area_intersect / area_union)

    return out_dict


def eval_pointcloud(pointcloud_pred, pointcloud_gt,
                    normals_pred=None, normals_gt=None):

    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        normals_gt, normals_pred
    )
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()


    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()


    # Chamfer distance
    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2

    if not normals_pred is None:
        accuracy_normals = accuracy_normals.mean()
        completeness_normals = completeness_normals.mean()
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    else:
        accuracy_normals = np.nan
        completeness_normals = np.nan
        normals_correctness = np.nan


    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
        'iou': np.nan
    }

    return out_dict


def distance_p2p(pointcloud_pred, pointcloud_gt,
                    normals_pred, normals_gt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(pointcloud_gt)
    dist, idx = kdtree.query(pointcloud_pred)

    if normals_pred is None:
        return dist, None

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    normals_dot_product = (normals_gt[idx] * normals_pred).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)

    return dist, normals_dot_product

def l1_loss(images_1, images_2, pixel_average=True):
    # expects (batch, h, w, c) inputs
    if images_1.ndim == 4: normalization = images_1.shape[0]*images_1.shape[1]*images_1.shape[2]*3
    elif images_1.ndim == 3: normalization = images_1.shape[0]*images_1.shape[1]*3
    
    channelwise_l1_loss = (np.abs(images_1[...,:3]-images_2[...,:3]))
    mean_l1_img_loss = np.sum(channelwise_l1_loss)/normalization
    if not pixel_average: return channelwise_l1_loss
    return mean_l1_img_loss

def l2_loss(images_1, images_2, pixel_average=True):
    # expects (batch, h, w, c) inputs
    if images_1.ndim == 4: normalization = images_1.shape[0]*images_1.shape[1]*images_1.shape[2]*3
    elif images_1.ndim == 3: normalization = images_1.shape[0]*images_1.shape[1]*3
    
    channelwise_l2_loss = np.sqrt( ((images_1[...,0] - images_2[...,0])**2) + 
                                   ((images_1[...,1] - images_2[...,1])**2) + 
                                   ((images_1[...,2] - images_2[...,2])**2)
                                 )
    if not pixel_average: return channelwise_l2_loss
    mean_L2_img_loss = np.sum(channelwise_l2_loss)/normalization
    return mean_L2_img_loss

def pixelwise_loss(predicted_mesh_path, gt_files_path, num_images = 3, pixel_average = True):
    elevation = np.arctan(0.6/1)/np.pi*180

    rnd_idx = rng.permutation(30)[:num_images]
    rendered_images = np.empty((num_images, 224,224, 3))
    prerendered_images = np.empty((num_images, 224,224, 3))

    for i, idx in enumerate(rnd_idx):
        image = render_mesh(predicted_mesh_path, elevation=elevation, start=idx)
        rendered_images[i] = image.cpu().numpy()[...,:3]
        
        sample_idx = str(idx*12).zfill(3)
        pyexr_img = pyexr.read(f'{gt_files_path}/_r_{sample_idx}.exr')
        pyexr_img[...,:3] = pyexr_img[...,:3]/3
        pyexr_img = rgba2rgb(pyexr_img)
        
        prerendered_images[i] = pyexr_img
    #print(rendered_images.shape, prerendered_images.shape)
    L1 = l1_loss(rendered_images, prerendered_images, pixel_average=pixel_average)
    L2 = l2_loss(rendered_images, prerendered_images, pixel_average=pixel_average)
    return L1, L2

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

if __name__ == "__main__":
    import glob
    import argparse
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    

    parser = argparse.ArgumentParser(
        description='Split Data'
    )

    parser.add_argument('--test', type=str, default='/media/alex/SSD Datastorage/guided-research/runs/occupancy/01091710_Ablation3_hybrid_singleconvx4_nofreeze/epoch=107-val_loss=0.6125.ckpt')
    parser.add_argument('--GT_path', type=str, default='/media/alex/SSD Datastorage/data/blender/car')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose')
    parser.add_argument('--pixelloss', dest='pixelloss', action='store_true', help='verbose')
    
    args = parser.parse_args()
    
    #prepare mesh_pathes to read and evaluate
    #point to experiment, read all folders, compare to folders of GT
    predicted_meshes_paths = glob.glob(args.test + '/*[!.txt]') #points to #mesh for predicted mesh
    exp_idx = [predicted_meshes_paths[i].split('/')[-1] for i in range(len(predicted_meshes_paths))]#[:50] #mesh
    GT_path = args.GT_path

    ###evaluation here  
    #define dict
    performance = {'completeness': [], 'accuracy': [],'normals completeness': [],'normals accuracy': [], 'normals': [], 'completeness2': [], 'accuracy2': [], 'chamfer_l2': [], 'iou': [], 'pixel_L1':[], 'pixel_L2':[]}
    names = []

    #evaluate meshes
    for i, idx in enumerate(tqdm(exp_idx)):
        #read meshes
        if args.verbose:
            print(f'evaluating mesh: {idx} at {predicted_meshes_paths[i]}')

        pred_mesh = trimesh.load(f'{predicted_meshes_paths[i]}/mesh.obj')
        gt_mesh = trimesh.load(f'{GT_path}/{idx}/disn_mesh.obj')
        out_dict = eval_mesh(pred_mesh, gt_mesh, -0.5, 0.5, n_points=100000)
        names.append(idx)
        
        #pixelwise loss
        L1, L2 = pixelwise_loss(f'{predicted_meshes_paths[i]}/mesh.obj', f'{GT_path}/{idx}')
        if (not np.any(np.isnan(L1))) and (not np.any(np.isnan(L2))): pass
        else: L1, L2 = 0, 0
        #L1, L2 = 0, 0      

        for key in out_dict.keys():
            performance[key].append(out_dict[key])
        performance['pixel_L1'].append(L1)
        performance['pixel_L2'].append(L2)
    
    #sort meshes according to iou for cherry picking and failure cases
    ious = performance['iou']
    sorted_names = [x for _,x in sorted(zip(ious,names))]
    sorted_ious = signif(sorted(ious), 3)

    #write file
    with open(f'{args.test}/results.txt', 'w') as file:
        n = len(performance['completeness'])
        file.write(str(n)+' meshes'+'\n')
        for key in performance.keys():
            file.write('mean '+key+': '+str(signif(np.sum(performance[key])/n,3))+'\n') #avg ('completeness'): ...
        file.write('\n')
        for key in performance.keys():
            file.writelines(key+":"+'\n'+str(signif(performance[key], 3)))
            file.write('\n')
            file.write('\n')
        file.write('sorted by best IoU')
        file.write('\n')
        for i in range(len(sorted_ious)):
            file.write(str(sorted_names[i])+": "+str(sorted_ious[i])+'\n')

