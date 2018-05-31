import os
import sys
import time
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append( 'util' )
from util import *
from model_def import *

use_gpu = True if torch.cuda.is_available() else False





num_workers = 4 if use_gpu else 0

BATCH_SIZE = 128
IMG_SIZE = 28

max_epoch = 50
learning_rate = 1e-4
weight_decay = 1e-5
patience = 10



retrain = True  # 是否重新训练，如果设置为False，需要额外设置dir_number和model_type

# 【注意】retrain为False时，模型结构必须相同
# 需要小心设置，本程序不对任何非法情况进行检查
if not retrain:
    dir_number = 0      # 选择文件夹编号，如'run_001'，dir_number = 1
    model_type = 'last' # model类型，last或best








directory = '../../BP4D+ Preprocess by PanBowen'  # 首选路径

if not os.path.exists( directory ): 
    directory = '/data2/BP4D+ Preprocess by PanBowen'   # 候选路径（因为173服务器上数据存放在/data2目录下）


metadata_dir = '%s/BP4D+ Save/Multimodal Dataset' % directory
visible_dir = '%s/BP4D+ Save/2DImages face' % directory
thermal_dir = '%s/BP4D+ Save/Thermal face' % directory








# 定义Dataset
train_set = MyDataset( metadata_dir, 'train', visible_dir, thermal_dir, IMG_SIZE )
train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers )

valid_set = MyDataset( metadata_dir, 'valid', visible_dir, thermal_dir, IMG_SIZE )
valid_loader = DataLoader( valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers )

test_set = MyDataset( metadata_dir, 'test', visible_dir, thermal_dir, IMG_SIZE )
test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers )

NUM_CLASSES = train_set.NUM_CLASSES








if retrain:
    model = CNNModel( 16 )    
else:
    # 载入已有模型
    model_path = 'output/run_%03d/model_%s.pkl' % ( dir_number, model_type )
    model = torch.load( model_path )


criterion = nn.CrossEntropyLoss()

if use_gpu:
    model.to('cuda')
    criterion.to('cuda')



optimizer = optim.Adam( model.parameters(), lr=learning_rate, weight_decay=weight_decay )
scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=patience, eps=1e-6, verbose=True )  # 学习率衰减








# 创建输出目录'run_XXX'
output_dir = 'output'

if not os.path.exists( output_dir ):
    os.makedirs( output_dir )

directory_list = os.listdir( output_dir )

if directory_list == []:
    number = 0    
else:
    number = int( directory_list[-1][-3:] ) + 1

output_path = '%s/run_%03d' % ( output_dir, number )

if not os.path.exists( output_path ):
    os.makedirs( output_path )








# 输出参数到文件中
filename = '%s/params.txt' % output_path

with open( filename, 'w' ) as f:
    print( '【retrain】', retrain, file=f )
    
    if not retrain:
        print( 'dir = \'run_%03d\', model_type = %s' % ( dir_number, model_type ), file=f )

    print( '【BATCH_SIZE】', BATCH_SIZE, file=f )
    print( '【IMG_SIZE】', IMG_SIZE, file=f )
    print( '【max_epoch】', max_epoch, file=f )
    print( '【optimizer】\n', optimizer, file=f )
    print( '【scheduler】 patience=%d, eps=%g' % ( scheduler.patience, scheduler.eps ), file=f )
    print( '【model】\n', model, file=f )
    
    print( '【model_def.py】', file=f )   # 记录model定义的代码
    model_def_filename = 'util/model_def.py'
    with open( model_def_filename, 'r' ) as f_model_def:
        print( f_model_def.read(), file=f )
    



print( '-------------------- Configuration --------------------' )
with open( filename, 'r' ) as f:
    print( f.read() )








print( '-------------------- Training --------------------' )
best_acc_valid = -1
columns=['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'test_loss', 'test_acc',
    'is_valid_better', 'learning_rate']
record = pd.DataFrame( [], columns=columns )



for epoch in range( max_epoch ):
 
    t0 = time.time()
    print( '\nEpoch = %d' % epoch )
    
    cur_learning_rate = optimizer.param_groups[0]['lr']
    print( 'learning_rate = %g' % cur_learning_rate )


    y_train = np.zeros( 0, dtype=int )
    pred_train = np.zeros( (0, NUM_CLASSES) )

    model.train()

    for batch_i, data in enumerate( train_loader, 0 ):
 
        batch_Xv, _, batch_y = data
 
        if use_gpu:
            batch_Xv, batch_y = batch_Xv.to('cuda'), batch_y.to('cuda')

    
        optimizer.zero_grad()
        output = model( batch_Xv )
        loss = criterion( output, batch_y )
        loss.backward()
        optimizer.step()


        batch_y = batch_y.to('cpu').numpy() if use_gpu else batch_y.numpy()
        output = output.to('cpu').detach().numpy() if use_gpu else output.detach().numpy()

        y_train = np.concatenate( [y_train, batch_y] )
        pred_train = np.concatenate( [pred_train, output] )


        
        if batch_i % 200 == 0:
            print( '\tbatch_i=%d\tloss=%f' % ( batch_i, loss.item() ) )





        # 用于非GPU服务器上DEBUG
        if not use_gpu: 
            if batch_i == 2: break





    loss_train, acc_train = compute_performance_visible( y_train, pred_train )
    print( 'Train\tloss=%f\tacc=%f\tin %.2f s' % ( loss_train, acc_train, time.time() - t0 ) )
    

    t0 = time.time()
    y_valid, pred_valid = model_predict_visible( model, valid_loader, use_gpu )
    loss_valid, acc_valid = compute_performance_visible( y_valid, pred_valid )
    print( 'Valid\tloss=%f\tacc=%f\tin %.2f s' % ( loss_valid, acc_valid, time.time() - t0 ), end='' )
    
    if acc_valid > best_acc_valid:
        print( '\tBetter Valid Performance!' )
    else:
        print()
    
    
    t0 = time.time()
    y_test, pred_test = model_predict_visible( model, test_loader, use_gpu )
    loss_test, acc_test = compute_performance_visible( y_test, pred_test )
    print( 'Test\tloss=%f\tacc=%f\tin %.2f s' % ( loss_test, acc_test, time.time() - t0 ) )






    # 保存当前epoch的训练结果
    values = [epoch, loss_train, acc_train, loss_valid, acc_valid, loss_test, acc_test,
        acc_valid > best_acc_valid, cur_learning_rate]
    temp = pd.DataFrame( [values], columns=columns )
    record = record.append( temp )

    filename = '%s/training_records.csv' % output_path
    record.to_csv( filename, index=False )






    # 保存当前epoch的模型
    filename = '%s/model_last.pkl' % output_path
    torch.save( model, filename )

    # 根据Valid Set的acc，保存best模型
    if acc_valid > best_acc_valid:
        best_acc_valid = acc_valid
        filename_best = '%s/model_best.pkl' % output_path
        shutil.copyfile( filename, filename_best )  # 复制文件





    # 根据Valid Set的acc，执行学习率衰减
    scheduler.step( acc_valid )




