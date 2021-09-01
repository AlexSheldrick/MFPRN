import argparse
from datetime import datetime
from pathlib import Path
from random import randint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=6, help='num workers')
    parser.add_argument('--gpu', type=int, nargs='+', default=0, help='gpus')
    parser.add_argument('--sanity_steps', type=int, default=1, help='overfit multiplier')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--splitsdir', type=str, default='blender', help='resume checkpoint')
    parser.add_argument('--datasetdir', type=str, help='datasetdir', default='../data')
    parser.add_argument('--val_check_percent', type=float, default=1.0, help='percentage of val checked')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='check val every fraction of epoch')
    parser.add_argument('--max_epoch', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
    parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
    
    
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    parser.add_argument('--res', type=int, default=64, help='Voxelsize for inference')
    parser.add_argument('--inf_res', type=int, default=1, help='Multiple of inference resolution per training grid resolution')
    parser.add_argument('--precision', type=int, default=32, help='float32 or float16 network precision')
    parser.add_argument('--profiler', type=str, default=None, help='Profiler: None, simple or Advanced')
    parser.add_argument('--version', type=str, default=None, help='version for logs name')
    parser.add_argument('--clamp', type=float, default=0.1, help='truncate sdf to certain value or convert to occupancy field')
    
    parser.add_argument('--encoder', type=str, default='hybrid', help='type of encoder')    #conv2d_pretrained_projective
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument("--coordinate_embedding_size", nargs='+', type=int, default=[256, 0], help='ndims for fourier embedding')
    parser.add_argument("--coordinate_embedding_scale", nargs='+', type=int, default=[0.8, 0], help='scale for fourier embedding')
    
    parser.add_argument('--fieldtype', type=str, default='occupancy', help='type of scalar field, sdf/occupancy/udf are possible values')
    parser.add_argument('--experiment', type=str, default='Ablation3_hybrid_full_nofreeze', help='experiment directory')
    parser.add_argument('--batch_size', type=int, default=7, help='batch size')
    
    parser.add_argument('--test', type=str, default=None, help='load and test from model-checkpoint')

    

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = randint(0, 999)

    if args.val_check_interval > 1:
        args.val_check_interval = int(args.val_check_interval)

    args.experiment = f"{datetime.now().strftime('%d%m%H%M')}_{args.experiment}"
    if args.resume is not None:
        args.experiment = Path(args.resume).parents[0].name

    return args
