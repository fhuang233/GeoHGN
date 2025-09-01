# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from models import GeoHGN
from datasets import GraphDataset, PLIDataLoader, DynamicBatchWrapper
from parms_setting import settings
import numpy as np
from utils import TrainLogger, set_seed
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from math import exp, pi, cos, log
#from torch.nn.parallel import DataParallel

import tempfile

#tempfile.tempdir = '../temp'

# %%

metric_heads = ['test2013_rmse', 'test2013_pr', 'test2016_rmse',
                'test2016_pr', 'test2019_rmse', 'test2019_pr']


def val(
    model, 
    dataloader, 
    device, 
    args,
):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(
                data, 
            )
            label = data.y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    
    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))


    return rmse, coff


def train(
    model, 
    dataloader, 
    loss_func, 
    optimizer, 
    device,
    scheduler,
    logger,
    args,
):  
  
    model.train()
    
    if args.max_n_vertex_per_gpu is not None:
       dataloader.dataset._form_batch()
       
    epoch_loss = 0.0
    total_num_samples = 0

    for data in dataloader:
        
        data = data.to(device)
        pred = model(
            data,  
        )
        
        
        label = data.y

        loss = loss_func(pred, label)
        
        update_flag = True
        if loss.isnan() or loss.isinf(): 
            logger.info(f'Current loss is {loss.item()}, do not update')
            update_flag = False

        if update_flag:
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            batch_num_samples = label.size(0)
            epoch_loss += loss.item() * batch_num_samples
            total_num_samples += batch_num_samples

        if args.sched_freq == 'batch':
            scheduler.step()
   
    
    if args.sched_freq == 'epoch':
        scheduler.step()
        
    epoch_loss = epoch_loss / (total_num_samples + 1e-12)

    epoch_rmse = np.sqrt(epoch_loss)           


    return epoch_loss, epoch_rmse


def get_dataloader(args, logger, dataset_name='train'):
    
    
    dataset_dir = os.path.join(args.data_root, dataset_name)
    dataset_df = pd.read_csv(os.path.join(args.data_root, dataset_name + '.csv'))
    graph_dataset = GraphDataset(dataset_dir, dataset_df, model_name='GeoHGN')
    
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    if args.max_n_vertex_per_gpu is not None:
        if args.valid_max_n_vertex_per_gpu is None:
            args.valid_max_n_vertex_per_gpu = args.max_n_vertex_per_gpu
        
        if dataset_name!='test':
            batch_size = 1
            num_workers = 1
        
        if dataset_name=='train':
            graph_dataset = DynamicBatchWrapper(graph_dataset, args.max_n_vertex_per_gpu)
            
            
        if dataset_name=='valid':
            graph_dataset = DynamicBatchWrapper(graph_dataset, args.valid_max_n_vertex_per_gpu)
    
    dataloader = PLIDataLoader(
        graph_dataset, 
        batch_size=batch_size, 
        shuffle=dataset_name=='train',
        #shuffle=False, 
        num_workers=num_workers)
    
    logger.info(f"{dataset_name} data: {len(graph_dataset)}")
    return dataloader
    

# %%
if __name__ == '__main__':
    
    args = settings()
    args.data_root = os.path.join(args.data_root, 'data')
    

    model_args = dict(
        cutoff_inter=args.cutoff_inter,
        cutoff_intra=args.cutoff_intra,
        num_conv_layers=args.num_conv_layers,
        hidden_channels=args.hidden_channels,
        middel_channels=args.middel_channels,
        num_radial_intra=args.num_radial_intra,
        num_spherical_intra=args.num_spherical_intra,
        num_radial_inter=args.num_radial_inter,
        num_spherical_inter=args.num_spherical_inter,
        num_res_layers=args.num_res_layers,
        num_trans_layers=args.num_trans_layers,
        num_output_layers=args.num_output_layers,
        num_dist_rbf_intra=args.num_dist_rbf_intra,
        num_dist_rbf_inter=args.num_dist_rbf_inter,
        num_heads=args.num_heads,
        act=args.act,
        attn_act=args.attn_act,
        pred_act=args.pred_act,
        proj_act=args.proj_act,
        use_drop=args.use_drop,
        use_bn=False if args.without_bn else True,
        use_gn=False if args.without_gn else True,
        use_res_in_proj=False if args.without_res else True,
        drop_rate=args.drop_rate,
        with_update_edge=args.with_update_edge,
        with_projector=False if args.without_projector else True,
        with_reslayer=False if args.without_reslayer else True,
        inter_att=True,
        intra_att=False,     
    )
    
    
    if torch.cuda.is_available():
        device = torch.device('cuda', args.cuda)
    else:
        device = torch.device('cpu')
    
    #set_seed(args.seed)
    
    results = defaultdict(list)
    for repeat in range(args.runs):
        args.repeat = repeat

        logger = TrainLogger(args)
        logger.info(__file__)
        
        train_loader = get_dataloader(args, logger, dataset_name='train')
        valid_loader = get_dataloader(args, logger, dataset_name='valid')
        test2013_loader = get_dataloader(args, logger, dataset_name='test2013')
        test2016_loader = get_dataloader(args, logger, dataset_name='test2016')
        test2019_loader = get_dataloader(args, logger, dataset_name='test2019')
        #testCSAR_loader = get_CSAR_dataloader(args)
        
        
        model = GeoHGN(35, **model_args)
        #model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loss_func = nn.MSELoss()
        
        scheduler = None
        if args.sched_freq is not None:
            step_per_epoch = (len(train_loader.dataset) + train_loader.batch_size - 1) // train_loader.batch_size
            max_step = args.epochs * step_per_epoch
            log_alpha = log(args.final_lr / args.lr) / max_step
            lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)   
        
        best_val_mse = float('inf')
        count = 0
        
        best_model_list = []
        
        for epoch in range(args.epochs):
            
            epoch_loss, epoch_rmse = train(
                model, 
                train_loader, 
                loss_func, 
                optimizer, 
                device,
                scheduler,
                logger,
                args,
            )
        
            # start validating
            valid_rmse, valid_pr = val(
                model, 
                valid_loader, 
                device, 
                args,
            )
            
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)

            if valid_rmse < best_val_mse:
                best_val_mse = valid_rmse
                count = 0
                if not args.no_save_model:
                    model_path = logger.save_model(model, msg)
                    best_model_list.append(model_path)
                    if len(best_model_list) > 5:
                        os.remove(best_model_list[0])
                        best_model_list.pop(0)
                    
            else:
                count = count + 1
                if count > args.early_stop_epoch:
                    msg = "best_val_mse: %.4f" % best_val_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break

        # final testing
        logger.save_best_model(best_model_list[-1])
        logger.load_model(model, best_model_list[-1])
        valid_rmse, valid_pr = val(model, valid_loader, device, args)
        test2013_rmse, test2013_pr = val(model, test2013_loader, device, args)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device, args)
        test2019_rmse, test2019_pr = val(model, test2019_loader, device, args)
        
        msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f," \
                    % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)

        logger.info(msg)
        
        if not args.no_logging_result:
        
                log_result = [{'Dataset': 'test2013', 'rmse': test2013_rmse, 'pr': test2013_pr}, 
                              {'Dataset': 'test2016', 'rmse': test2016_rmse, 'pr': test2016_pr}, 
                              {'Dataset': 'test2019', 'rmse': test2019_rmse, 'pr': test2019_pr}]
                
                logger.save_result(log_result)
        
        for key in metric_heads:
            
            results[key].append(eval(key))
    
    
    for key, value in results.items():
        print(key + ':' + str(value))
    
    
    results_df = pd.DataFrame(results)
    result_statistics = results_df.describe()
    for head in metric_heads:
        res_sta = result_statistics[head]
        print("%s: %.3f (%.3f)" % (head, res_sta.iloc[1], res_sta.iloc[2]))
        
            
            
            
            
        
        
# %%