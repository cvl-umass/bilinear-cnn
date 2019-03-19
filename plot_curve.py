import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse


res = [re.compile('Iteration (\d+)/(\d+)'),
        re.compile('Train Loss: ([.\d]+) Acc: ([.\d]+)'), 
        re.compile('Validation Loss: ([.\d]+) Acc: ([.\d]+)')]

def plot_acc(log_name):

    data = {}
    with open(log_name) as f:
        lines = f.readlines()
    for l in lines:
        i = 0
        for r in res:
            m = r.match(l)
            if m is not None:
                break
            i += 1
        if m is None:
            continue
        if i == 0:
            iteration = int(m.groups()[0])
            total_iter = int(m.groups()[1])
            if iteration not in data:
                data[iteration] = [0] * 4
        else:
            loss = float(m.groups()[0])
            acc = float(m.groups()[1])

            data[iteration][(i-1)*2] += loss
            data[iteration][(i-1)*2 +1] += acc

    
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    for k, v in data.items():
        train_loss.append(v[0])
        train_acc.append(v[1])
        val_loss.append(v[2])
        val_acc.append(v[3])


    iter_list = [int(x) + 1 for x in data.keys()]
    x_train = iter_list
    x_val = iter_list
    # x_train = np.arange(len(train_acc))
    # x_val = np.arange(len(val_acc))
    plt.subplot(1, 2, 1)
    plt.plot(x_train, train_acc, '-', linestyle='-', color='r', linewidth=2,
            label='train_top1')
    plt.plot(x_val, val_acc, '-', linestyle='-', color='b', linewidth=2,
            label='val_top1')
    plt.legend(loc="best")
    plt.xticks(np.arange(0, iter_list[-1], iter_list[-1]//10))
    plt.yticks(np.arange(0.1, 1, 0.05))
    plt.xlim([0, iter_list[-1]])
    # plt.ylim([min([min(train_acc), min(val_acc)]),
    #             max([max(train_acc), max(val_acc)])])
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(x_train, train_loss, '-', linestyle='-', color='r', linewidth=2,
            label='train_loss')
    plt.semilogy(x_val, val_loss, '-', linestyle='-', color='b', linewidth=2,
            label='val_loss')
    plt.legend(loc="best")
    plt.xticks(np.arange(0, iter_list[-1], iter_list[-1]//10))
    plt.yticks(np.arange(0.1, 1, 0.05))
    plt.xlim([0, iter_list[-1]])
    # plt.ylim([min([min(train_loss), min(val_loss)]),
    #             max([max(train_loss), max(val_loss)])])
    plt.yscale('log')
    plt.grid(True)

    return max(val_acc)


def plot_log(log_path, save_path, close_fig=True):
    plt.figure(figsize=(14, 8))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")

    max_acc = plot_acc(log_path)
    plt.grid(True)
    plt.savefig(save_path)
    if close_fig:
        plt.close()

    return max_acc

def main(args):
    _ = plot_log(os.path.join('augmentation_exp', args.exp_dir, args.logs),
            os.path.join('augmentation_exp', args.exp_dir, args.output_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=str, default='train_history.txt')
    parser.add_argument('--exp_dir', type=str, default='exp')
    parser.add_argument('--output_filename', type=str,
            default='training_curve.png')

    args = parser.parse_args()
    main(args)
