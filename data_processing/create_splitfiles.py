import argparse
import glob
from pathlib import Path
import random

# Create split files given percentages for each subset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create split files'
    )

    parser.add_argument('--train_percentage', type=float, default=0.85)
    parser.add_argument('--val_percentage', type=float, default=0.1)
    parser.add_argument('--test_percentage', type=float, default=0.05)
    parser.add_argument('--target_path', type=str, default="../data/splits/blender_full_fix/")

    parser.add_argument('--dataset_path', type=str, default="../data/blender/car")

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    target_path = Path(args.target_path)
    train_percentage = args.train_percentage
    val_percentage = args.val_percentage
    test_percentage = args.test_percentage    

    files = glob.glob(str(dataset_path / '*' / 'disn_mesh.obj'))
    banlist = [131, 220, 634, 704, 854, 835, 804, 862, 865, 1090, 1941, 1979, 2483, 3109]
    for idx in banlist:
        for i, filepath in enumerate(files):
            if str(idx) in filepath:
                files.pop(i)

    num_samples = len(files)

    splitsdir = {'train':[], 'val':[], 'test':[]}

    random.shuffle(files)
    
    a, b, c = int(train_percentage*num_samples), int(val_percentage*num_samples), int(test_percentage*num_samples)

    splitsdir['train'] = files[:a]
    splitsdir['val'] = files[a:a+b]
    splitsdir['test'] = files[a+b:a+b+c]
    #splitsdir['test'] = splitsdir['val'][:10]
    
    target_path.mkdir(exist_ok=True, parents=True)
    
    for split in splitsdir.keys():
        with open(str(target_path / f"{split}.txt"), 'w') as split_file:
            splitsdir[split] = '\n'.join(sorted(splitsdir[split])).replace('/disn_mesh.obj','')
            split_file.writelines(splitsdir[split])
