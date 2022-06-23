import argparse
from datetime import datetime
from pathlib import Path
from random import randint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=3, help='num workers')
    parser.add_argument('--gpu', type=int, nargs='+', default=0, help='gpus')
    parser.add_argument('--sanity_steps', type=int, default=0, help='overfit multiplier')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
    #global 2d/3d
    #parser.add_argument('--resume', type=str, default='/media/alex/SSD Datastorage/guided-research/runs/sdf/26051900_Chibane_sdf_ResNet-1C80-NoAugs-FIXDense32Recon/checkpoints/epoch=18-val_loss=0.1246.ckpt', help='resume checkpoint')
    #pixel aligned
    #parser.add_argument('--resume', type=str, default='/media/alex/SSD Datastorage/guided-research/runs/sdf/23052055_Chibane_sdf_ResNet-1C80-NoAugs-FIXDense32Recon/checkpoints/epoch=18-val_loss=0.0674.ckpt')
    #pixel aligned #2
    #parser.add_argument('--resume', type=str, default='/media/alex/SSD Datastorage/guided-research/runs/sdf/03062155_Chibane_sdf_ResNet-1C80-NoAugs-FIXDense32Recon/checkpoints/epoch=19-val_loss=0.0672.ckpt')
    
    parser.add_argument('--splitsdir', type=str, default='chibane_shapenet', help='resume checkpoint')
    #parser.add_argument('--splitsdir', type=str, default='overfit', help='resume checkpoint')
    
    parser.add_argument('--datasetdir', type=str, help='datasetdir', default='../data')
    parser.add_argument('--val_check_percent', type=float, default=1.0, help='percentage of val checked')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='check val every fraction of epoch')
    parser.add_argument('--max_epoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e-4, help='learning rate decay')

    parser.add_argument('--seed', type=int, default=45, help='random seed') #45

    parser.add_argument('--res', type=int, default=64, help='Voxelsize for inference')
    parser.add_argument('--inf_res', type=int, default=1, help='Multiple of inference resolution per training grid resolution')
    parser.add_argument('--precision', type=int, default=32, help='float32 or float16 network precision')
    parser.add_argument('--profiler', type=str, default='simple', help='Profiler: None, simple or Advanced')
    parser.add_argument('--version', type=str, default=None, help='version for logs name')
    parser.add_argument('--clamp', type=float, default=2, help='truncate sdf to certain value or convert to occupancy field')
    
    parser.add_argument('--encoder', type=str, default='hybrid_depthproject', help='type of encoder')    #conv2d_pretrained_projective #hybrid_depthproject #hybrid_surface
    parser.add_argument('--num_points', type=int, default=4000)
    parser.add_argument('--num_surface_points', type=int, default=2000)
    
    parser.add_argument("--coordinate_embedding_size", nargs='+', type=int, default=[256, 0], help='ndims for fourier embedding') #256 0 ##best 256
    parser.add_argument("--coordinate_embedding_scale", nargs='+', type=int, default=[1.75, 0], help='scale for fourier embedding') #0.8 0 ##best 1.75
    
    #important stuff
    parser.add_argument('--fieldtype', type=str, default='sdf', help='type of scalar field, sdf/occupancy/udf are possible values')
    parser.add_argument('--experiment', type=str, default='Chibane_sdf_ResNet-1C80-NoAugs-FIXDense32Recon', help='experiment directory')
    parser.add_argument('--batch_size', type=int, default=14, help='batch size') #11
    parser.add_argument('--freeze_pretrained', type=int, default=None, help='freeze_pretrained weights up to layer n')

    parser.add_argument("--transforms", nargs='+', type=str, default='None', help='types of transforms') #['Brightness', 'ColorJitter', 'ColorPermute', 'HorizontalFlip']
    parser.add_argument('--aug_threshhold', type=float, default=0.5, help='1-p threshhold probability to reach to apply random augmentations')
    parser.add_argument('--implicit_grad', type=int, default=None, help='Change forward pass to extract gradients of implicit SDF')
    parser.add_argument('--autoLR', type=int, default=None, help='Change forward pass to extract gradients of implicit SDF')
    
    parser.add_argument('--num_voxels', type=int, default=32, help='Voxelsize')
    parser.add_argument('--voxel_type', type=str, default='colored_density', help='Options are occupancy, density, colored_density')
    
    parser.add_argument('--test', type=str, default=None, help='load and test from model-checkpoint')
    parser.add_argument('--min_z', type=float, default= -1.4041, help='minimum sdf value for entire dataset')
    parser.add_argument('--max_z', type=float, default=  1.6160, help='max sdf value for entire dataset')
    
    

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = randint(0, 999)

    if args.val_check_interval > 1:
        args.val_check_interval = int(args.val_check_interval)

    args.experiment = f"{datetime.now().strftime('%d%m%H%M')}_{args.experiment}"
    if args.resume is not None:
        args.experiment = Path(args.resume).parents[1].name

    return args
