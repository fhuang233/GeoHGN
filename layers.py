from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GraphNorm
import torch.nn as nn
from geo_emb import dist_emb
import torch.nn.functional as F
from model_utils import CosineCutoff
from torch_scatter import scatter
from utils import check_nan


class Predictor(nn.Module):
    def __init__(
        self, 
        dim_graph_emb, 
        dim_pred_emb, 
        num_layers, 
        num_tasks, 
        act=nn.LeakyReLU(),
        use_dropout=True, 
        use_bn=True, 
        **kwargs
    ):
        super(Predictor, self).__init__()
        
        if use_dropout:
        
            self.dprate = 0.1 if 'drop_rate' not in kwargs else kwargs['drop_rate']
        
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.num_layers = num_layers
        self.act = act
        
        self.lins = nn.ModuleList()
        for j in range(num_layers):
            if j == 0:
                self.lins.append(nn.Linear(dim_graph_emb, dim_pred_emb))
            elif j == num_layers - 1:
                self.lins.append(nn.Linear(dim_pred_emb, num_tasks))
            else:
                self.lins.append(nn.Linear(dim_pred_emb, dim_pred_emb))

      
        if use_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(dim_pred_emb) for _ in range(num_layers - 1)])


    def forward(self, h):
        for i, lin in enumerate(self.lins):
            if i < self.num_layers - 1:
                h = self.act(lin(h))
                if self.use_dropout:
                    h = F.dropout(h, self.dprate, training=self.training)
                if self.use_bn:
                    h = self.bns[i](h)
            else:
                h = lin(h)   

        return h


class Projector(nn.Module):
    def __init__(
        self, 
        hidden_channels, 
        num_layers, 
        act=nn.SiLU(), 
        use_gn=True, 
        res=True
    ):
        super(Projector, self).__init__()
        
        self.act = act
        self.res = res
        self.use_gn = use_gn
        self.lins = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)])
             
        if use_gn:
            self.gns = nn.ModuleList([GraphNorm(hidden_channels) for _ in range(num_layers)])


    def forward(self, h, batch):
    
        for i, lin in enumerate(self.lins):
        
            temp = self.act(lin(h))
            if self.res:
                h = temp + h 
            if self.use_gn:
                h = self.gns[i](h, batch) 
            
        return h


class InteractionLayer(nn.Module):
    def __init__(
            self,
            last_layer=False,
            **kwargs
    ):
        super(InteractionLayer, self).__init__()
        
        if last_layer:
            kwargs['with_update_edge'] = False
        
        hete_convs = [HeteConv(**kwargs) for _ in range(2)]
        
        self.hete_conv_l = hete_convs[0]
        self.hete_conv_p = hete_convs[1]


    def forward(
        self,
        x_l,
        x_p,  
        edge_attr_l, 
        edge_attr_p, 
        edge_attr_l2p, 
        edge_attr_p2l,
        batch_l,
        batch_p, 
        edge_index,
        sphe_emb,
        torsion_emb,
        dist,
        ):
        
        h_l, edge_attr_l, edge_attr_p2l = self.hete_conv_l(
                                              x_p,
                                              x_l,
                                              edge_attr_l,
                                              edge_attr_p2l,
                                              edge_index['l'],
                                              edge_index['p2l'],
                                              sphe_emb['l'],
                                              torsion_emb['l'],
                                              sphe_emb['p2l'],
                                              torsion_emb['p2l'],
                                              dist['l'],
                                              dist['p2l'],
                                              batch_l,
                                          )
        
        h_p, edge_attr_p, edge_attr_l2p = self.hete_conv_p(
                                              x_l,
                                              x_p,
                                              edge_attr_p,
                                              edge_attr_l2p,
                                              edge_index['p'],
                                              edge_index['l2p'],
                                              sphe_emb['p'],
                                              torsion_emb['p'],
                                              sphe_emb['l2p'],
                                              torsion_emb['l2p'],
                                              dist['p'],
                                              dist['l2p'],
                                              batch_p,
                                          )
        
        return h_l, h_p, edge_attr_l, edge_attr_p, edge_attr_l2p, edge_attr_p2l


class HeteConv(nn.Module):
    
    def __init__(
        self,
        middle_channels=64,
        hidden_channels=256,
        num_radial_intra=3,
        num_spherical_intra=2,
        num_radial_inter=3,
        num_spherical_inter=2,
        cutoff_intra=0.8,
        cutoff_inter=0.5,
        num_res_layers=3,
        num_heads=8,
        act=nn.SiLU(),
        attn_act=nn.SiLU(),
        with_update_edge=True,
        with_reslayer=True,
        inter_att=True,
        intra_att=False,
    ):
    
        super(HeteConv, self).__init__()        
        
        self.with_reslayer = with_reslayer
        self.act = act
               
        IntraConv_kwargs = dict(
            middle_channels=middle_channels,
            hidden_channels=hidden_channels,
            num_radial=num_radial_intra,
            num_spherical=num_spherical_intra,
            cutoff=cutoff_intra,
            act=act,
            with_update_edge=with_update_edge,
            conv_type='intra',
        )
        
        InterConv_kwargs = dict(
            middle_channels=middle_channels,
            hidden_channels=hidden_channels,
            num_radial=num_radial_inter,
            num_spherical=num_spherical_inter,
            cutoff=cutoff_inter,
            act=act,
            with_update_edge=with_update_edge,
            conv_type='inter'
        )
        IntraConv = ConvLayer
        InterConv = ConvLayer
        
        if intra_att:
            IntraConv_kwargs['attn_act'] = attn_act
            IntraConv_kwargs['num_heads'] = num_heads
            IntraConv = ConvLayerAtt
        if inter_att:
            InterConv_kwargs['attn_act'] = attn_act
            InterConv_kwargs['num_heads'] = num_heads
            InterConv = ConvLayerAtt

        self.conv_t = IntraConv(**IntraConv_kwargs)
        self.conv_s2t = InterConv(**InterConv_kwargs)

        self.cat_proj = nn.Linear(hidden_channels * 2, hidden_channels)
        
        if with_reslayer:
            self.lins = torch.nn.ModuleList()
            for _ in range(num_res_layers):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.gn = GraphNorm(hidden_channels)
        
    def forward(
        self,
        x_s,
        x_t,
        edge_attr_t,
        edge_attr_s2t,
        edge_index_t,
        edge_index_s2t,
        sphe_emb_t,
        torsion_emb_t,
        sphe_emb_s2t,
        torsion_emb_s2t,
        dist_t,
        dist_s2t,
        batch,   
    ):
        
        h_t_homo, edge_attr_t = self.conv_t(
                                    x_s=x_t,
                                    x_t=x_t,
                                    edge_index=edge_index_t,
                                    edge_attr=edge_attr_t,
                                    sphe_emb=sphe_emb_t,
                                    torsion_emb=torsion_emb_t,
                                    dist=dist_t
                                )
        
        h_t_hete, edge_attr_s2t = self.conv_s2t(
                                      x_s=x_s,
                                      x_t=x_t,
                                      edge_index=edge_index_s2t,
                                      edge_attr=edge_attr_s2t,
                                      sphe_emb=sphe_emb_s2t,
                                      torsion_emb=torsion_emb_s2t,
                                      dist=dist_s2t
                                  )
        
        h_t = torch.cat([h_t_homo, h_t_hete], 1)
        h_t = self.cat_proj(h_t)
        h_t = self.act(h_t)
        
        x_t = h_t + x_t
        
        if self.with_reslayer:
            for i, lin in enumerate(self.lins):
                x_t = self.act(lin(x_t)) + x_t
        
        x_t = self.gn(x_t, batch)
        
        return x_t, edge_attr_t, edge_attr_s2t         


class BaseConv(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        with_update_edge=True,
        act=nn.SiLU(),  
        aggr="add", 
        node_dim=0
    ):
        super(BaseConv, self).__init__(aggr=aggr, node_dim=node_dim)
        
        self.hidden_channels = hidden_channels
        self.with_update_edge = with_update_edge
        self.act = act
        
        if with_update_edge:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 3)
            #self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.cat_f_proj = nn.Linear(hidden_channels * 2, hidden_channels)
            #self.dd_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        
        
    def edge_update(self, f_ji, d1_ji, d2_ji):
        '''
        if self.with_update_edge:
            f1, f2, f3 = torch.split(self.act(self.f_proj(f_ji)), self.hidden_channels, dim=1)
            dd = self.dd_proj(d_msg_i)
            f_ji = f1 + self.act(self.cat_f_proj(torch.cat([f2 * d1_ji, f3 * d2_ji], 1))) * dd
            
            return f_ji
        else:
            return f_ji
        '''
        if self.with_update_edge:
            f1, f2, f3 = torch.split(self.act(self.f_proj(f_ji)), self.hidden_channels, dim=1)
            #dd = self.dd_proj(d_msg_i)
            f_ji = f1 + self.act(self.cat_f_proj(torch.cat([f2 * d1_ji, f3 * d2_ji], 1)))
            
            return f_ji
        else:
            return f_ji
    
    
    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_msg, d_msg = features
        
        s_msg = scatter(s_msg, index, dim=self.node_dim, dim_size=dim_size)
        d_msg = scatter(d_msg, index, dim=self.node_dim, dim_size=dim_size)
        
        return s_msg, d_msg
    
    
    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class ConvLayer(BaseConv):
    def __init__(
        self,
        middle_channels: int,
        hidden_channels: int,
        num_radial: int,
        num_spherical: int,
        cutoff=8.0,
        act=nn.SiLU(),
        conv_type='intra',
        **kwargs, 
    ):
        super(ConvLayer, self).__init__(hidden_channels, act=act, **kwargs)
        
        self.conv_type = conv_type
        
        self.sphe_proj = nn.Sequential(
            nn.Linear(num_radial * num_spherical ** 2, middle_channels, bias=False),
            nn.Linear(middle_channels, hidden_channels, bias=False))
        self.torsion_proj = nn.Sequential(
            nn.Linear(num_radial * num_spherical, middle_channels, bias=False),
            nn.Linear(middle_channels, hidden_channels, bias=False))
            
        self.node_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
        self.s_msg_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        self.cat_msg_proj = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)
        
        self.o_msg_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if conv_type == 'intra':
            self.final_proj = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.final_proj = nn.Linear(hidden_channels*2, hidden_channels)
        
        self.cutoff = CosineCutoff(cutoff)
    
    
    def forward(
        self, 
        x_s,
        x_t, 
        edge_index, 
        edge_attr, 
        sphe_emb, 
        torsion_emb, 
        dist
    ):
        h_s = x_s
        h_t = x_t
        d1 = self.sphe_proj(sphe_emb)
        d2 = self.torsion_proj(torsion_emb)
        
        s_msg, d_msg = self.propagate(
            edge_index,
            v=self.node_proj(h_s),
            f_ji=edge_attr,
            d1_ji=d1,
            d2_ji=d2,
            r_ji=dist,
            size=(h_s.size(0), h_t.size(0)),
        )
        
        d_msg = self.cat_msg_proj(d_msg)   
        f_ji = self.edge_updater(edge_index, f_ji=edge_attr, d1_ji=d1, d2_ji=d2)
        
        s_msg1, s_msg2 = torch.split(self.o_msg_proj(s_msg), self.hidden_channels, dim=1)
        
        if self.conv_type == 'intra':
            h_t = self.act(self.final_proj(h_t + s_msg1 + s_msg2 * d_msg))
        else:
            h_t = self.act(self.final_proj(torch.cat([h_t, s_msg1 + s_msg2 * d_msg], 1)))
        
        return h_t, f_ji


    def message(self, v_j, f_ji, d1_ji, d2_ji, r_ji):
        
        dv = self.act(self.dv_proj(f_ji)) * self.cutoff(r_ji).view(-1, 1)
        # dv = self.act(self.dv_proj(f_ji))
        v_j = v_j * dv
        
        s1, s2 = torch.split(self.act(self.s_msg_proj(v_j)), self.hidden_channels, dim=1)
        d_ji = torch.cat([s1 * d1_ji, s2 * d2_ji], 1)

        return v_j, d_ji


class ConvLayerAtt(BaseConv):
    def __init__(
        self,
        middle_channels: int,
        hidden_channels: int,
        num_radial: int,
        num_spherical: int,
        num_heads: int,
        cutoff=5.0,
        act=nn.SiLU(),
        attn_act=nn.SiLU(),
        conv_type='intra',
        **kwargs, 
        ):
        super(ConvLayerAtt, self).__init__(hidden_channels, act=act, **kwargs)

        self.conv_type = conv_type
        
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        
        self.sphe_proj = nn.Sequential(
            nn.Linear(num_radial * num_spherical ** 2, middle_channels, bias=False),
            nn.Linear(middle_channels, hidden_channels, bias=False))
        self.torsion_proj = nn.Sequential(
            nn.Linear(num_radial * num_spherical, middle_channels, bias=False),
            nn.Linear(middle_channels, hidden_channels, bias=False))
        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
        self.o_att = nn.Linear(hidden_channels, hidden_channels)
        
        self.s_msg_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        self.cat_msg_proj = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)
        
        self.o_msg_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        
        if conv_type == 'intra':
            self.final_proj = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.final_proj = nn.Linear(hidden_channels*2, hidden_channels)
        
        self.cutoff = CosineCutoff(cutoff)

    def forward(
        self,  
        x_s, 
        x_t, 
        edge_index, 
        edge_attr, 
        sphe_emb, 
        torsion_emb, 
        dist
    ):
        
        h_s = x_s
        h_t = x_t
        
        
        d1 = self.sphe_proj(sphe_emb)
        d2 = self.torsion_proj(torsion_emb)
        
        q = self.q_proj(h_t).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(h_s).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(h_s).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(edge_attr)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(edge_attr)).reshape(-1, self.num_heads, self.head_dim)
        
        s_msg, d_msg = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            d1_ji=d1,
            d2_ji=d2,
            r_ji=dist,
            size=(h_s.size(0), h_t.size(0)),
        )
        
        d_msg = self.cat_msg_proj(d_msg)
      
        f_ji = self.edge_updater(edge_index, f_ji=edge_attr, d1_ji=d1, d2_ji=d2)
        
        s_msg1, s_msg2 = torch.split(self.o_msg_proj(s_msg), self.hidden_channels, dim=1)
        
        if self.conv_type == 'intra':
            h_t = self.act(self.final_proj(h_t + s_msg1 + s_msg2 * d_msg))
        else:
            h_t = self.act(self.final_proj(torch.cat([h_t, s_msg1 + s_msg2 * d_msg], 1)))
        
        return h_t, f_ji


    def message(self, q_i, k_j, v_j, dk, dv, d1_ji, d2_ji, r_ji):
        
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.act(attn) * self.cutoff(r_ji).unsqueeze(1)
        
        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)
        v_j = self.act(self.o_att(v_j))
        
        s1, s2 = torch.split(self.act(self.s_msg_proj(v_j)), self.hidden_channels, dim=1)
        d_ji = torch.cat([s1 * d1_ji, s2 * d2_ji], 1)

        return v_j, d_ji


