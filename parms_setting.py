import argparse


def settings():
    parser = argparse.ArgumentParser()
    
    # config
    parser.add_argument('--model_name', type=str, default='GeoHGN')
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='./ckpts')
    parser.add_argument('--runs', type=int, default=3)
    
    parser.add_argument('--split_sixty', action='store_true', default=False, 
                        help='use split by sequence identity 60 for LBA')
                        
    parser.add_argument('--split_type', type=str, default='sequence_identity',
                        choices=['sequence_identity', 'random', 'scaffold'], 
                        help='split type used for cold start experiment')
    parser.add_argument('--final_lr', type=float, default=1e-6, help='final learning rate')
    parser.add_argument('--max_n_vertex_per_gpu', type=int, default=None, help='if specified, ignore batch_size and form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--valid_max_n_vertex_per_gpu', type=int, default=None, help='form batch with dynamic size constrained by the total number of vertexes')
    parser.add_argument('--sched_freq', type=str, default=None, 
                        help='scheduler mode')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients with too big norm')
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--early_stop_epoch', type=int, default=100)
    parser.add_argument('--no_save_model', action='store_true', default=False)
    parser.add_argument('--no_logging_result', action='store_true', default=False)
    parser.add_argument('--cuda', type=int, default=0,
                        help='gpu_id. Default is 0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_mark', type=str, default='')
    
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--wd', type=float, default=5e-6, help='weight decay (L2 loss on parameters).')
    
    # model args           
                       
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed. Default is 42.')
                       
    parser.add_argument('--cutoff_inter', type=float, default=5.0)
    parser.add_argument('--cutoff_intra', type=float, default=6.0)
    parser.add_argument('--num_conv_layers', type=int, default=6,
                       help='number of interaction layers, default is 6.')
    
    parser.add_argument('--hidden_channels', type=int, default=256,
                       help='dimensionality of hidden layers, default is 256.')
    
    parser.add_argument('--middel_channels', type=int, default=64,
                       help='dimensionality of middel layer in geometric linear, default is 64.')
            
    parser.add_argument('--num_radial_intra', type=int, default=3,)
    parser.add_argument('--num_spherical_intra', type=int, default=2,)
    parser.add_argument('--num_radial_inter', type=int, default=3,)
    parser.add_argument('--num_spherical_inter', type=int, default=2,)
    
    parser.add_argument('--num_res_layers', type=int, default=4,
                       help='number of linear layers in resnet within the conv layer, default is 4.')
    
    parser.add_argument('--num_trans_layers', type=int, default=4,
                       help='number of linear layers in projectors followed by the interaction block, default is 4.')
    
    parser.add_argument('--num_output_layers', type=int, default=3,
                       help='number of linear layers in predictor, default is 3.')
    
    parser.add_argument('--num_dist_rbf_intra', type=int, default=16,
                       help='dimensionality of dist rbf intra net, default is 16.')
    
    parser.add_argument('--num_dist_rbf_inter', type=int, default=16,
                       help='dimensionality of dist rbf inter net, default is 16.')
    
    parser.add_argument('--num_heads', type=int, default=8,
                       help='number of heads in attention net, default is 8.')        
    
    parser.add_argument('--act', type=str,
                        choices=['ssp', 'silu', 'tanh', 'sigmoid', 'swish', 'leakyrelu', 'relu', 'prelu', 'elu', 'softmax', 'selu'],
                        default='silu')
    parser.add_argument('--attn_act', type=str,
                        choices=['ssp', 'silu', 'tanh', 'sigmoid', 'swish', 'leakyrelu', 'relu', 'prelu', 'elu', 'softmax', 'selu'],
                        default='silu')
    parser.add_argument('--pred_act', type=str,
                        choices=['ssp', 'silu', 'tanh', 'sigmoid', 'swish', 'leakyrelu', 'relu', 'prelu', 'elu', 'softmax', 'selu'],
                        default='silu') 
    parser.add_argument('--proj_act', type=str,
                        choices=['ssp', 'silu', 'tanh', 'sigmoid', 'swish', 'leakyrelu', 'relu', 'prelu', 'elu', 'softmax', 'selu'],
                        default='silu')
     
    parser.add_argument('--use_drop', action='store_true', default=False, 
                        help='use dropout in predictor, default=False')
    
    parser.add_argument('--without_bn', action='store_true', default=False, 
                        help='not use batchnorm in predictor, default=False')
    
    parser.add_argument('--without_gn', action='store_true', default=False, 
                        help='not use graphnorm in projector, default=False')
    
    parser.add_argument('--without_res', action='store_true', default=False, 
                        help='not use residual connection in projector, default=False')
    
    parser.add_argument('--with_update_edge', action='store_true', default=False, 
                        help='update edge attribution, default=False')
    
    parser.add_argument('--without_projector', action='store_true', default=False, 
                        help='not use projector, default=False')
                        
    parser.add_argument('--without_reslayer', action='store_true', default=False, 
                        help='not use residual in convlayer, default=False')
                        
    parser.add_argument('--geo_comenet', action='store_true', default=False, 
                        help='use comenet geometries, default=False')
    
    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability) in predictor. Default is 0.5.')
    
    
    args = parser.parse_args()

    return args