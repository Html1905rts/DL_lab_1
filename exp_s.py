#
# Хисматуллин Владимир 
# модуль для проведения экспериментов и отрисовки графиков.
#


from turtle import width
import numpy as np
import pandas as pd
from regex import W
import torch
import torch.nn as nn
from utils import get_dataloaders, FFN, fit, get_distribution_over_layers, act_N
import torch.nn.functional as F
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

def exp_activations(train_data_loader, test_data_loader, activation, act_name, title, color, n_classes=1):
    """
    Эксперимент для сравнения функций активации
    activation - сама функция
    act_name - её название 
    title - заголовок для графика
    """
    num_features = next(iter(train_data_loader))[0].shape[1]
    layers_list = [num_features ] + list(np.zeros(4, dtype=int) + 40) + [n_classes]
    act = [activation] * len(layers_list)
    model = FFN(layers_list, act) 

    
    opt = torch.optim.Adagrad(model.parameters(), lr=5e-3)
    
    if n_classes == 1:
        loss = torch.nn.MSELoss()
        loss_name = 'MSE loss'
    else:
        loss = torch.nn.CrossEntropyLoss()
        loss_name = 'Cross Entropy loss'

    history = fit(10, model, loss, opt,
        train_data_loader, test_loader=test_data_loader, hist_idx=[9])
    print(act_name + ' ' + loss_name + ' : {:.4f}'.format(history[0][1].detach().numpy()))
    
    hist = get_distribution_over_layers(model, test_data_loader) 
    fig, _ = plt.subplots(2, 4 ,figsize=(4*3.5, 2*3))
    for i in range(8):

        ax = plt.subplot(2, 4, i + 1)
        if i not in [0, 4]:
            ax.set_ylabel(' ')
        else:
            ax.set_ylabel('Плотность', fontsize=12)

        if i % 2 == 0:
            ax.set_xlabel('Линейный слой № ' + str(i // 2 + 1), fontsize=12)
            sns.histplot(hist[0][i // 2], bins=50, stat="density", color=color,kde = True, ax=ax)
        else:
            ax.set_xlabel('Слой '+ act_name  + ' № ' + str(i // 2 + 1), fontsize=12)
            sns.histplot(hist[1][i // 2], bins=50, stat="density", color=color, kde = True, ax=ax)

        ax.grid(True)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.suptitle(title, fontsize=14)
    plt.show()




def exp_FFN(train_data_loader, test_data_loader, title, model_strings, n_models=1, n_classes=1, mode=None,
            width=30, BN_list=None, DO_list=None, skip_list=None, clip_list=None, init_lists=None):
    """
    Универсальная функция для экспериментов с фиксированной глубиной в 8 слоёв:
    (8 линейных + 8 BN + 8 DO + 8 активации)
    
    mode - режим показа: 
        'small' -    активации на 2, 4, 6, 8 линейном слое 
        'both'-      акивации на линейных слоях и после нелинейности на 2, 4, 6, 8 слоях
        'fns' -      акивации  после нелинейности на на всех 8 слоях
        'linear' - активации на всех 8 линейных слоях
        
        #'large' - слоёв становится 16, показываются все чётные

    """
    num_features = next(iter(train_data_loader))[0].shape[1]
    
    depth = 8

    if isinstance(width, int): 
        layers_list = [ 
        [num_features ] + list(np.zeros(depth, dtype=int) + width) + [n_classes] for i in range(n_models)
        ]
    elif isinstance(width[0], int) :
        layers_list = [ 
        [num_features ] + list(np.zeros(depth, dtype=int) + width_) + [n_classes] for width_ in width
        ]
    else :
        layers_list = [
            [num_features] + list(layers) + [n_classes] for layers in width
        ]
    
    if clip_list is None:
        act = [[nn.GELU()] * len(layers_list[0])] * n_models
    else:
        act = [[act_N(nn.GELU(), clip)] * len(layers_list[0]) for clip in clip_list]

    if BN_list is None:
        BN_list = [None] * n_models
    if DO_list is None:
        DO_list = [None] * n_models
    if skip_list is None:
        skip_list = [None] * n_models
    if init_lists is None:
        init_lists = [None] * n_models

    models = [FFN(layers_list[i], act[i], batch_norm_list=BN_list[i], drop_out_list=DO_list[i], 
                init_list=init_lists[i], skip_list=skip_list[i]) for i in range(n_models)]

    if n_classes == 1:
        loss = torch.nn.MSELoss()
        loss_name = 'MSE loss'
    else:
        loss = torch.nn.CrossEntropyLoss()
        loss_name = 'Cross Entropy loss'

    opt_s = [torch.optim.Adagrad(models[i].parameters(), lr=5e-3) for i in range(n_models)]

    for i in range(n_models):
        history = fit(30, models[i], loss, opt_s[i],
            train_data_loader, test_loader=test_data_loader, hist_idx=[29])
        print(model_strings[i] + ' : ' + loss_name + ' = ', np.around(float(history[0][1]), 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = [get_distribution_over_layers(models[i], test_data_loader) for i in range(n_models)]

    colors = ['C'+str(i) for i in range(n_models)]
    
    if mode is None or mode == 'small':
        plt.subplots(1, 4 ,figsize=(16, 10))
        for i in range(4):
            ax = plt.subplot(2, 4, i + 1)
            if i not in [0, 4]:
                ax.set_ylabel(' ')
            else:
                ax.set_ylabel('Плотность', fontsize=14)
                
            quant_1 =  10000
            quant_2 =  -10000

            for j in range(n_models):
                # Выбираем минимальную из квантилей
                quant_1 = np.min([quant_1, np.quantile(hist[j][0][(i + 1) * 2 - 1], 0.01)])
                quant_2 = np.max([quant_2, np.quantile(hist[j][0][(i + 1) * 2 - 1], 0.99)])
                sns.kdeplot(hist[j][0][(i + 1) * 2 - 1], color=colors[j], ax=ax, linewidth = 2, alpha=0.8, bw_adjust=0.4)
                ax.set_title('Линейный слой № ' + str((i + 1) * 2), fontsize=14)
                ax.set_xlim(quant_1, quant_2)
            ax.grid(True)
            if i == 2:
                ax.legend([string for string in model_strings], loc='upper center', 
             bbox_to_anchor=(-0.2, -0.1),fancybox=False, shadow=False, ncol=n_models, fontsize=14)

    else:
        plt.subplots(2, 4 ,figsize=(16, 10))
        for i in range(8):
            ax = plt.subplot(2, 4, i + 1)
            if i not in [0, 4]:
                ax.set_ylabel(' ')
            else:
                ax.set_ylabel('Плотность', fontsize=14)
            
            quant_1 =  10000
            quant_2 =  -10000

            if mode == 'both':
                for j in range(n_models):
                    if i % 2 == 0:
                        # Слой 2, 4, 6, 8
                        quant_1 = np.min([quant_1, np.quantile(hist[j][0][(i // 2 + 1) * 2 - 1], 0.01)])
                        quant_2 = np.max([quant_2, np.quantile(hist[j][0][(i // 2 + 1) * 2 - 1], 0.99)])
                        sns.kdeplot(hist[j][0][(i // 2 + 1) * 2 - 1], color=colors[j], ax=ax, linewidth = 2, alpha=0.8, bw_adjust=0.4)
                        ax.set_title('Линейный слой № ' + str((i // 2 + 1) * 2), fontsize=14)
                    else:
                        quant_1 = np.min([quant_1, np.quantile(hist[j][1][(i // 2 + 1) * 2 - 1], 0.01)])
                        quant_2 = np.max([quant_2, np.quantile(hist[j][1][(i // 2 + 1) * 2 - 1], 0.99)])
                        sns.kdeplot(hist[j][1][(i // 2 + 1) * 2 - 1], color=colors[j], ax=ax, linewidth = 2, alpha=0.8, bw_adjust=0.4)
                        ax.set_title('Нелинейность № ' + str((i // 2 + 1) * 2), fontsize=14)
                ax.set_xlim(quant_1, quant_2)

            elif mode == 'fns':
                for j in range(n_models):
                    quant_1 = np.min([quant_1, np.quantile(hist[j][1][i], 0.01)])
                    quant_2 = np.max([quant_2, np.quantile(hist[j][1][i], 0.99)])
                    sns.kdeplot(hist[j][1][i], color=colors[j], ax=ax, linewidth = 2, alpha=0.8, bw_adjust=0.4)
                    ax.set_title('Слой нелинейности № ' + str(i+1), fontsize=14)
                ax.set_xlim(quant_1, quant_2)

            else:
                for j in range(n_models):
                    quant_1 = np.min([quant_1, np.quantile(hist[j][0][i], 0.01)])
                    quant_2 = np.max([quant_2, np.quantile(hist[j][0][i], 0.99)])
                    sns.kdeplot(hist[j][0][i], color=colors[j], ax=ax, linewidth = 2, alpha=0.8, bw_adjust=0.4)
                    ax.set_title('Линейный слой № ' + str(i+1), fontsize=14)
                ax.set_xlim(quant_1, quant_2)
            ax.grid(True)
            if i == 6:
                ax.legend([string for string in model_strings], loc='upper center', 
             bbox_to_anchor=(-0.2, -0.1),fancybox=False, shadow=False, ncol=n_models, fontsize=14)

    plt.suptitle(title, fontsize=14)
    plt.show()