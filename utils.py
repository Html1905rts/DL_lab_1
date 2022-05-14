#
# Хисматуллин Владимир 
# модуль с реализацией нейронных сетей и вещей с ними связанных
#

import warnings
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes, load_digits
from sklearn.preprocessing import StandardScaler

def get_dataloaders(dataset_name, test_size, batch_size):
    """

    dataset_name - str - название датасета из sklearn toy

    Информация о датасетах:

    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    << Name           |  Type     ||  Instances  ||  Features  >>
    <<--------------------------------------------------------->>
    << boston         |  REG      ||  506        ||  13        >>
    << breast_cancer  |  CLF, 2   ||  569        ||  30        >>
    << diabetes       |  REG      ||  442        ||  10        >>
    << digits         |  CLF, 10  ||  1797       ||  64        >>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    """
    if dataset_name == 'boston':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = load_boston()['data']
            y = load_boston()['target']
            y = StandardScaler().fit_transform(y.reshape(-1, 1))
            y_type = torch.float32

    elif dataset_name == 'breast_cancer':
        X = load_breast_cancer()['data']
        y = load_breast_cancer()['target']
        y_type = torch.long 

    elif dataset_name == 'diabetes':
        X = load_diabetes()['data']
        y = load_diabetes()['target']
        y = StandardScaler().fit_transform(y.reshape(-1, 1))
        y_type = torch.float32
        

    elif dataset_name == 'digits':
        X = load_digits()['data']
        y = load_digits()['target']
        y_type = torch.long 
    
    # Стандартизация
    X = StandardScaler().fit_transform(X)

    # Разбиение на трейн, тест + создание датасетов
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).to(y_type))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).to(y_type))
    
    return (DataLoader(train_data, batch_size=batch_size),  
            DataLoader(test_data, batch_size=batch_size) )



class FFN(torch.nn.Module):
    def __init__(self, layers_dim_list, act_fn_list, 
    batch_norm_list = None, drop_out_list=None, skip_list=None, init_list=None):
        """
        Нейронная сеть 
        
        layers_dims - list of integers: n-ое число - размер n слоя.
            ( 0-й элемент - размер входного слоя )
            
        
        act_fns     - list of activations: n-ая функция - функция активации n+1 слоя
        
        batch_norm_list - list of bool n-ое значение - есть после n-го слоя BN или нет

        drop_out_list - list of reals: n-ое число либо параметр DO, либо 0, если слоя нет

        skip_list - для каждого слоя указывается слой с которого нужно прокинуть связь

        init_list - list of tuples: (function, kwargs)

        Для сети с входом 20, тремя внутренними слоями по 30, 35 и 40, выходом в 2 числа,
        а так же DO после первого, BN после второго слоя и активациями RELU,
        skip-connect выхода первого на третий  
        
        нужно передать [20, 30, 35, 40, 2], [RELU, RELU, RELU], [0, 1, 0], [0.2, 0, 0], [-1, -1, 1]

        """
        super(FFN, self).__init__()
        
        self.depth = len(layers_dim_list) - 2

        if batch_norm_list is None:
            batch_norm_list = [0] * self.depth
        
        if drop_out_list is None:
            drop_out_list = [0] * self.depth
    
        if skip_list is None:
            self.skip_list = [-1] + [-1] * self.depth
        else:
            self.skip_list = skip_list
        
        if init_list is None:
            init_list = [(nn.init.xavier_uniform_ , {})] * (self.depth + 1)

        # Учитываем skip-connect в ширине слоёв
        prev_layer = layers_dim_list.copy()
        for i in range(1, self.depth + 1):
            if self.skip_list[i] > -1:
                prev_layer[i] += prev_layer[self.skip_list[i]]

        self.activations_after_linear = [0] * self.depth 
        self.activations_after_layer = [0] * self.depth 
        # Способ хранения модели не классический, но вполне удобный
        self.act_fns = []
        self.layers = []
        self.BN = []
        self.DO = []
        
        
        # По глубине добавляем все нужные слои
        for i in range(self.depth):
            self.layers.append(torch.nn.Linear(prev_layer[i], layers_dim_list[i+1]))
            init_list[i][0](self.layers[i].weight, **init_list[i][1])
            if batch_norm_list[i]:
                self.BN.append(torch.nn.BatchNorm1d(layers_dim_list[i+1]))
            else:
                self.BN.append(nn.Identity())
            if drop_out_list[i] < 1.e-6:
                self.DO.append(nn.Identity())
            else:
                self.DO.append(nn.Dropout(drop_out_list[i]))
            
            self.act_fns.append(act_fn_list[i])

        self.layers.append(torch.nn.Linear(prev_layer[self.depth], layers_dim_list[self.depth+1]))
        init_list[self.depth][0](self.layers[self.depth].weight, **init_list[self.depth][1])

 
    def forward(self, input):
        # Во-первых мы сохраняем активации для skip-connect
        # Во-вторых они нам нужны по задаче
        x = [0] * (self.depth+1)
        x[0] = input
        for i in range(self.depth):
            # Линейный слой 
            x[i+1] = self.layers[i](x[i])
            # Запоминаем показания после линейного пре-ия
            self.activations_after_linear[i] = x[i+1] 

            # Batch Norm
            x[i+1] = self.BN[i](x[i+1])
            # Drop Out
            x[i+1] = self.DO[i](x[i+1])
            
            # Активация
            x[i+1] = self.act_fns[i](x[i+1])

            # Возможно skip-connect
            if self.skip_list[i+1] != -1:
                x[i+1] = torch.cat([x[i+1], x[self.skip_list[i+1]]], 1)
            # Запоминаем показания после слоя
            self.activations_after_layer[i] = x[i+1] 

        return self.layers[-1](x[-1])
    
    def get_activations(self, input=None):
        """
        Возвращает активации для каждого элемента батча
        
        Если input не None, то делает прямой проход
        """
        if input is not None:
            self.forward(input)

        return self.activations_after_linear, self.activations_after_layer
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.weight)
            params.append(layer.bias)
        return params

def get_distribution_over_layers(model, dataloader):
    """
    Возвращает распределение по слоям
    
    returns: (linear_layer_distribution, activation_layer_distribution)
    linear_layer_distribution: np.array of shape(n_layers, n_objects * layer_width)
    """
    distribution_after_linear = [[] for i in range(model.depth)]
    distribution_after_layer = [[] for i in range(model.depth)]
    model.eval()

    for X, y in dataloader:
        activations = model.get_activations(X)
        acts_linear = activations[0]
        acts_layer = activations[1]

        for ind in range(len(acts_linear)):
            distribution_after_linear[ind] = distribution_after_linear[ind] +\
                                            list(acts_linear[ind].detach().numpy().flatten())

            distribution_after_layer[ind] = distribution_after_layer[ind] +\
                                            list(acts_layer[ind].detach().numpy().flatten())

    return (np.array([np.array(arr) for arr in distribution_after_linear]),
            np.array([np.array(arr) for arr in distribution_after_layer])   )


def evaluate(dataloader, model, loss_fn):
    model.eval()
    
    total_loss = 0.0
    len = 0
    with torch.no_grad():
        for X, y in dataloader:
            # 2. Perform forward pass
            preds = model(X) 
            
            # 3. Evaluate loss
            total_loss += loss_fn(preds, y)
            len += y.shape[0]
        
    return total_loss / len

def fit(num_epochs, model, loss_fn, opt, train_dl, test_loader=None, hist_idx=None, get_info_fn=None):
    """
    hist_idx - эпохи"""
    if hist_idx is None:
        hist_idx = np.arange(1, num_epochs + 1)
    history = []

    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0
        len = 0

        for X, y in train_dl: 
            # Прямой ход
            pred = model(X) 
            
            # Считаем loss.
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            len += y.shape[0]
            
            # Град. спуск.
            opt.zero_grad() 
            loss.backward() 
            opt.step()

        # Заполняем статистики
        if (epoch+1) in hist_idx:
            if test_loader is None:
                if get_info_fn is not None:
                    history.append((total_loss / len, evaluate(train_dl, model, get_info_fn)))
                else:
                    history.append(total_loss / len)
            else:
                if get_info_fn is not None:
                    history.append((total_loss / len, evaluate(test_loader, model, loss_fn), evaluate(test_loader, model, get_info_fn)))
                else:
                    history.append((total_loss / len, evaluate(test_loader, model, loss_fn)))
    
    return history

def act_N(torch_actication_fn, upper_bound):
    """
    Создаёт функцию активации, обрезанную сверху числом N
    """
    if upper_bound is None:
        def inner_activation(x):
            acts = torch_actication_fn(x)
            return  acts
    else:
        def inner_activation(x):
            acts = torch_actication_fn(x)
            return  acts * (acts < upper_bound) + upper_bound * (acts >= upper_bound)
    return inner_activation