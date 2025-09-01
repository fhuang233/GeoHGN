# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from models import GeoHGN
from datasets import GraphDataset, PLIDataLoader
from parms_setting import settings
import numpy as np
from utils import TrainLogger, set_seed
from sklearn.metrics import mean_squared_error
from collections import defaultdict
#from torch.nn.parallel import DataParallel

# %%


def val(
    model, 
    dataloader, 
    device, 
    #epoch=-1
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
    
    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))


    return rmse, coff


def train(
    model, 
    dataloader, 
    loss_func, 
    optimizer, 
    device, 
    #epoch=-1
):
    model.train()
            
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
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        
        batch_num_samples = label.size(0)
        epoch_loss += loss.item() * batch_num_samples
        total_num_samples += batch_num_samples
        
    epoch_loss = epoch_loss / (total_num_samples + 1e-12)

    epoch_rmse = np.sqrt(epoch_loss)           
    
    

    return epoch_loss, epoch_rmse


def get_dataloader(args, logger, dataset_name='train'):
    
    dataset_dir = os.path.join(args.data_root, 'data', 'train')
    dataset_df = pd.read_csv(os.path.join(args.data_root, 'cold_start_data', f'{dataset_name}_{args.split_type}.csv'))
    graph_dataset = GraphDataset(dataset_dir, dataset_df, model_name='GeoHGN')
    dataloader = PLIDataLoader(
        graph_dataset, 
        batch_size=args.batch_size, 
        shuffle=dataset_name=='train',
        #shuffle=False, 
        num_workers=4)
    logger.info("cold start experiment")
    logger.info(f"{dataset_name} data: {len(graph_dataset)}")
    return dataloader

    

# %%
if __name__ == '__main__':
    
    args = settings()
    
    args.save_dir = os.path.join(args.save_dir, 'cold_start_' + args.split_type)
    
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
        test_loader = get_dataloader(args, logger, dataset_name='test')
        
        
        model = GeoHGN(35, **model_args)
        #model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loss_func = nn.MSELoss()
        
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
                #epoch
            )
            
            # start validating
            valid_rmse, valid_pr = val(
                model, 
                valid_loader, 
                device, 
                #epoch
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
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test_rmse, test_pr = val(model, test_loader, device)

        
        msg = "valid_rmse-%.4f, valid_pr-%.4f, test_rmse-%.4f, test_pr-%.4f" \
                    % (valid_rmse, valid_pr, test_rmse, test_pr)

        logger.info(msg)
        
        log_result = {'rmse': test_rmse, 'pearson': test_pr}
        
        if not args.no_logging_result:
                
                logger.save_result([log_result], fieldnames=[_ for _ in log_result])
        
        for key in log_result:
            
            results[key].append(log_result[key])
    
    
    for key, value in results.items():
        print('split by ' + args.split_type)
        print(key + ':' + str(value))
    
    
    results_df = pd.DataFrame(results)
    result_statistics = results_df.describe()
    for head in log_result:
        res_sta = result_statistics[head]
        print("%s: %.3f (%.3f)" % (head, res_sta.iloc[1], res_sta.iloc[2]))
        
            
            
            
            
        
        
# %%