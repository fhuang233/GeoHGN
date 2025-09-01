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
from scipy.stats import pearsonr, spearmanr
from math import exp, pi, cos, log
#from torch.nn.parallel import DataParallel

# %%


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
                #epoch
            )
            label = data.y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    
    pearson_res = pearsonr(pred, label)
    spearman_res = spearmanr(pred, label)
    rmse = np.sqrt(mean_squared_error(label, pred))


    return rmse, pearson_res, spearman_res


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
        #print(data.idx)
        data = data.to(device)
        pred = model(
            data, 
            #epoch
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


def get_LBA_dataloader(args, logger, dataset_name='train'):
     
    dataset_dir = os.path.join(args.data_root, 'graph_data')

    split_root = os.path.join(args.data_root, args.split_info)
        
    dataset_df = pd.read_csv(os.path.join(split_root, dataset_name + '.csv'))
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
        batch_size=args.batch_size, 
        shuffle=dataset_name=='train',
        #shuffle=False, 
        num_workers=4)
    
    
    logger.info(f"{dataset_name} data: {len(graph_dataset)}")
    return dataloader
    

# %%
if __name__ == '__main__':
    
    args = settings()
    
    args.data_root = os.path.join(args.data_root, 'LBA_data')
    
    if args.split_sixty:
        args.split_info = 'split-by-sequence-identity-60'
    else:
        args.split_info = 'split-by-sequence-identity-30'
    
    args.save_dir = os.path.join(args.save_dir, 'LBA_'+args.split_info)

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
        intra_att=True,      
    )
    
    if torch.cuda.is_available():
        device = torch.device('cuda', args.cuda)
    else:
        device = torch.device('cpu')
    
    seeds = [2023, 2022, 2021]  # 3 parallel experiments as suggested by previous baselines
    
    results = defaultdict(list)
    for repeat, seed in enumerate(seeds):
        
        set_seed(seed)
        args.repeat = repeat

        logger = TrainLogger(args)
        logger.info(__file__)
        logger.info(args.split_info)
        
        
        model = GeoHGN(35, **model_args)
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            
        loss_func = nn.MSELoss()
 
        train_loader = get_LBA_dataloader(args, logger, dataset_name='train')
        valid_loader = get_LBA_dataloader(args, logger, dataset_name='val')
        test_loader = get_LBA_dataloader(args, logger, dataset_name='test')
        
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
            valid_rmse, valid_pr, valid_sp = val(
                model, 
                valid_loader, 
                device, 
                args,
            )
            
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr.statistic)
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
        valid_rmse, valid_pr, valid_sp = val(model, valid_loader, device, args)
        test_rmse, test_pr, test_sp = val(model, test_loader, device, args)
        
        msg = "valid_rmse-%.4f, valid_pr-%.4f, valid_sp-%.4f, test_rmse-%.4f, test_pr-%.4f, test_sp-%.4f," \
                    % (valid_rmse, valid_pr.statistic, valid_sp.statistic, test_rmse, test_pr.statistic, test_sp.statistic)

        logger.info(msg)
        
        log_result = {'rmse': test_rmse, 'pearson': test_pr.statistic, 'pearson_pvalue': test_pr.pvalue, 
                               'spearman': test_sp.statistic, 'spearman_pvalue': test_sp.pvalue
                             }
        
        if not args.no_logging_result:    
            logger.save_result([log_result], fieldnames=[_ for _ in log_result])
        
        for key in log_result:
            
            results[key].append(log_result[key])
    
    
    for key, value in results.items():
        print(key + ':' + str(value))
    
    
    results_df = pd.DataFrame(results)
    result_statistics = results_df.describe()
    for head in log_result:
        res_sta = result_statistics[head]
        print("%s: %.3f (%.3f)" % (head, res_sta.iloc[1], res_sta.iloc[2]))
        
            
            
            
            
        
        
# %%