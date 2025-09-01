import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
import math
from geo_emb import angle_emb, sphe_emb
import copy

from model_utils import (
    CosineCutoff, 
    ExpNormalSmearing, 
    GaussianSmearing, 
    NeighborEmbedding, 
    EdgeEmbedding,
    act_func,
)

from layers import (
    Predictor,
    Projector, 
    InteractionLayer,
)

from geo_utils import (
    calculate_geo_intra,
    calculate_geo_inter,   
)


class GeoHGN(nn.Module):
    def __init__(
            self,
            node_dim=35, 
            cutoff_inter=5.0,
            cutoff_intra=5.0,
            num_conv_layers=6,
            hidden_channels=256,
            middel_channels=64,
            out_channels=1,
            num_radial_intra=3,
            num_spherical_intra=2,
            num_radial_inter=3,
            num_spherical_inter=2,
            num_res_layers=3,
            num_trans_layers=3,
            num_output_layers=3,
            num_dist_rbf_intra=32,
            num_dist_rbf_inter=32,
            num_heads=8,
            act='silu',
            attn_act='silu',
            pred_act='silu',
            proj_act='silu',
            use_drop=False,
            use_bn=True,
            use_gn=True,
            use_res_in_proj=True,
            drop_rate=0.1,
            with_update_edge=True,
            with_projector=True,
            with_reslayer=True,
            inter_att=True,
            intra_att=False,  
    ):
        super(GeoHGN, self).__init__()
        
        
        self.cutoff_inter = cutoff_inter
        self.cutoff_intra = cutoff_intra
        
        self.rbf_dist_l = ExpNormalSmearing(cutoff_intra, num_dist_rbf_intra)
        self.rbf_dist_p = ExpNormalSmearing(cutoff_intra, num_dist_rbf_intra)
        self.rbf_dist_l2p = ExpNormalSmearing(cutoff_inter, num_dist_rbf_inter)
        self.rbf_dist_p2l = ExpNormalSmearing(cutoff_inter, num_dist_rbf_inter)
        
        self.edge_embedding_l = EdgeEmbedding(num_dist_rbf_intra, hidden_channels, edge_type='intra')
        self.edge_embedding_p = EdgeEmbedding(num_dist_rbf_intra, hidden_channels, edge_type='intra')
        self.edge_embedding_l2p = EdgeEmbedding(num_dist_rbf_inter, hidden_channels, edge_type='inter')
        self.edge_embedding_p2l = EdgeEmbedding(num_dist_rbf_inter, hidden_channels, edge_type='inter')
        
        self.node_embedding_l = NeighborEmbedding(node_dim, hidden_channels, num_dist_rbf_intra, 
                                    num_dist_rbf_inter, cutoff_intra, cutoff_inter)
        self.node_embedding_p = NeighborEmbedding(node_dim, hidden_channels, num_dist_rbf_intra, 
                                    num_dist_rbf_inter, cutoff_intra, cutoff_inter)
        
        self.sphe_emb_intra = sphe_emb(num_radial=num_radial_intra, num_spherical=num_spherical_intra, cutoff=cutoff_intra)
        self.torsion_emb_intra = angle_emb(num_radial=num_radial_intra, num_spherical=num_spherical_intra, cutoff=cutoff_intra)
        
        self.sphe_emb_inter = sphe_emb(num_radial=num_radial_inter, num_spherical=num_spherical_inter, cutoff=cutoff_inter)
        self.torsion_emb_inter = angle_emb(num_radial=num_radial_inter, num_spherical=num_spherical_inter, cutoff=cutoff_inter)
        
        self.output_block = OutputBlock(
            hidden_channels=hidden_channels,
            middle_channels=hidden_channels,
            out_channels=out_channels,
            num_trans_layers=num_trans_layers,
            num_output_layers=num_output_layers,
            pred_act=pred_act,
            proj_act=proj_act,
            use_drop=use_drop,
            use_bn=use_bn,
            use_gn=use_gn,
            use_res_in_proj=use_res_in_proj,
            drop_rate=drop_rate,
            with_projector=with_projector,
        )
    
        ConvLayer_kwargs = dict(
            middle_channels=middel_channels,
            hidden_channels=hidden_channels,
            num_radial_intra=num_radial_intra,
            num_spherical_intra=num_spherical_intra,
            num_radial_inter=num_radial_inter,
            num_spherical_inter=num_spherical_inter,
            num_res_layers=num_res_layers,
            cutoff_intra=cutoff_intra,
            cutoff_inter=cutoff_inter,
            num_heads=num_heads,
            act=act_func[act](),
            attn_act=act_func[attn_act](),
            with_update_edge=with_update_edge,
            with_reslayer=with_reslayer,
            inter_att=inter_att,
            intra_att=intra_att,
        )
                  
        self.interaction_blocks = torch.nn.ModuleList([InteractionLayer(**ConvLayer_kwargs,
                    last_layer=False,) for _ in range(num_conv_layers - 1)])
        
        self.interaction_blocks.append(InteractionLayer(**ConvLayer_kwargs, last_layer=True))
    
    def forward(
        self, 
        data, 
    ):
        batch_l, batch_p = data['ligand'].batch, data['pocket'].batch
        (x_l, x_p, edge_attr_l, edge_attr_p, 
            edge_attr_l2p, edge_attr_p2l, conv_func_kwargs) = self.get_conv_kwargs(data)
        
        for i, interaction_block in enumerate(self.interaction_blocks):
            x_l, x_p, edge_attr_l, \
                edge_attr_p, edge_attr_l2p, \
                    edge_attr_p2l = interaction_block(
                                        x_l, 
                                        x_p, 
                                        edge_attr_l, 
                                        edge_attr_p, 
                                        edge_attr_l2p, 
                                        edge_attr_p2l,
                                        batch_l,
                                        batch_p, 
                                        **conv_func_kwargs)
        
        x = self.output_block(xs=(x_l, x_p), batchs=(batch_l, batch_p))

        return x


    def get_conv_kwargs(self, data):
        
        f_l, pos_l = data['ligand'].x, data['ligand'].pos
        f_p, pos_p = data['pocket'].x, data['pocket'].pos
        edge_index_l = data['ligand', 'intra_l', 'ligand'].edge_index
        edge_index_p = data['pocket', 'intra_p', 'pocket'].edge_index
        edge_index_l2p = data['ligand', 'inter_l2p', 'pocket'].edge_index
        edge_index_p2l = data['pocket', 'inter_p2l', 'ligand'].edge_index
        
        (dist_l, theta_l, phi_l, tau_l,
            dist_p, theta_p, phi_p, tau_p,
                dist_l2p, theta_l2p, phi_l2p, tau_l2p,
                    dist_p2l, theta_p2l, phi_p2l, tau_p2l) = self.get_geo(
                        edge_index_l, edge_index_p, edge_index_l2p, edge_index_p2l, pos_l, pos_p)
        
        
        rbf_l = self.rbf_dist_l(dist_l) 
        rbf_p = self.rbf_dist_p(dist_p)
        rbf_l2p = self.rbf_dist_l2p(dist_l2p)
        rbf_p2l = self.rbf_dist_p2l(dist_p2l)
        
        x_l = self.node_embedding_l(f_p, f_l, edge_index_l, edge_index_p2l, dist_l, dist_p2l, rbf_l, rbf_p2l)
        
        x_p = self.node_embedding_p(f_l, f_p, edge_index_p, edge_index_l2p, dist_p, dist_l2p, rbf_p, rbf_l2p)
        
        edge_attr_l = self.edge_embedding_l(edge_index_l, rbf_l, x_l, x_l)
        
        edge_attr_p = self.edge_embedding_p(edge_index_p, rbf_p, x_p, x_p)
        
        edge_attr_l2p = self.edge_embedding_l2p(edge_index_l2p, rbf_l2p, x_l, x_p)

        edge_attr_p2l = self.edge_embedding_p2l(edge_index_p2l, rbf_p2l, x_p, x_l)
        
        conv_func_kwargs = dict(
            edge_index={
                'l': edge_index_l,
                'p': edge_index_p,
                'l2p': edge_index_l2p,
                'p2l': edge_index_p2l
            },
            sphe_emb={
                'l': self.sphe_emb_intra(dist_l, theta_l, phi_l),
                'p': self.sphe_emb_intra(dist_p, theta_p, phi_p),
                'l2p': self.sphe_emb_inter(dist_l2p, theta_l2p, phi_l2p),
                'p2l': self.sphe_emb_inter(dist_p2l, theta_p2l, phi_p2l)
            },
            torsion_emb={
                'l': self.torsion_emb_intra(dist_l, tau_l),
                'p': self.torsion_emb_intra(dist_p, tau_p),
                'l2p': self.torsion_emb_inter(dist_l2p, tau_l2p),
                'p2l': self.torsion_emb_inter(dist_p2l, tau_p2l) 
            },
            dist={
                'l': dist_l,
                'p': dist_p,
                'l2p': dist_l2p,
                'p2l': dist_p2l 
            },
        )
        return x_l, x_p, edge_attr_l, edge_attr_p, edge_attr_l2p, edge_attr_p2l, conv_func_kwargs
        
                
    def get_geo(
            self, 
            edge_index_l, 
            edge_index_p, 
            edge_index_l2p, 
            edge_index_p2l, 
            pos_l, 
            pos_p
        ):
        
        vecs_l, dist_l, theta_l, phi_l, \
            tau_l, argmin0_l, argmin1_l = calculate_geo_intra(edge_index_l, pos_l)
        vecs_p, dist_p, theta_p, phi_p, \
            tau_p, argmin0_p, argmin1_p = calculate_geo_intra(edge_index_p, pos_p)
        
        dist_l2p, theta_l2p, phi_l2p, tau_l2p, \
            dist_p2l, theta_p2l, phi_p2l, tau_p2l = \
                calculate_geo_inter(
                    vecs_l, 
                    argmin0_l, 
                    argmin1_l, 
                    vecs_p, 
                    argmin0_p, 
                    argmin1_p, 
                    pos_l, 
                    pos_p, 
                    edge_index_l2p, 
                    edge_index_p2l
                )
        

        return (
            dist_l, theta_l, phi_l, tau_l,
            dist_p, theta_p, phi_p, tau_p,
            dist_l2p, theta_l2p, phi_l2p, tau_l2p,
            dist_p2l, theta_p2l, phi_p2l, tau_p2l
        )
    
    
class OutputBlock(nn.Module):
    def __init__(
            self,
            hidden_channels=256,
            middle_channels=256,
            out_channels=1,
            num_trans_layers=3,
            num_output_layers=3,
            pred_act='silu',
            proj_act='silu',
            use_drop=False,
            use_bn=True,
            use_gn=True,
            use_res_in_proj=True,
            drop_rate=0.1,
            with_projector=True,   
    ):
        super(OutputBlock, self).__init__()
        
        self.with_projector = with_projector
        if with_projector:
            self.trans_l = Projector(hidden_channels, num_trans_layers, act=act_func[proj_act](), use_gn=use_gn, res=use_res_in_proj)
            self.trans_p = Projector(hidden_channels, num_trans_layers, act=act_func[proj_act](), use_gn=use_gn, res=use_res_in_proj)
        
        self.predictor = Predictor(hidden_channels, middle_channels, num_output_layers, out_channels, 
                                      act=act_func[pred_act](), use_dropout=use_drop, use_bn=use_bn, drop_rate=drop_rate)
    
    def forward(
        self, 
        xs,
        batchs,
        edge_attrs=None, 
    ):
        batch_l, batch_p = batchs
        if edge_attrs is not None:
            edge_attr_l, edge_attr_p = edge_attrs
        
        x_l, x_p = xs   
        
        if self.with_projector:
            x_l = self.trans_l(x_l, batch_l)
            x_p = self.trans_p(x_p, batch_p)
        
        x = global_add_pool(x_l, batch_l) + global_add_pool(x_p, batch_p)
        
        x = self.predictor(x)

        return x.view(-1)



