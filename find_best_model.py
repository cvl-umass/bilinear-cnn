import argparse
import os
import re

def main(args):
    folder_path = os.path.join('..', 'exp', args.dataset)
    res = re.compile('Best val accuracy: ([.\d]+)')
    best_acc = 0
    best_model = None

    for folder in os.listdir(folder_path):
        train_hist_file = os.path.join(folder_path, folder, 'train_history.txt')
        if not os.path.isfile(train_hist_file):
            break
        with open(train_hist_file, 'r') as f :
            lines = f.readlines()
            for l in lines:
                if 'Best val accuracy:' in l:
                    m = res.match(l)
                    acc = float(m.groups()[0])
        if best_acc < acc:
            best_acc = acc
            best_model = folder

    print('Best model folder: %s \n Best accuracy: %f\n'%(best_model, best_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cub', type=str)
    args = parser.parse_args()
    
    main(args)
