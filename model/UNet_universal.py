import torch
import torch.nn as nn
from .point_convolution_universal import TransitionDown, TransitionUp
from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet


class UNet(nn.Module):
    def __init__(self, feat_dims: list, up_feat_dims: list, n_samples: list, n_layers: int, in_feat_dim: int, map_feat_dim=None,
                 need_feat_map: bool = False, k: int=16, radius=None):
        super(UNet, self).__init__()
        self.need_feat_map = need_feat_map

        self.feat_dims = [in_feat_dim] + feat_dims
        self.n_samples = n_samples
        self.n_layers = n_layers
        self.k = k

        self.up_feat_dims = up_feat_dims
        cur_up_feat_dims = up_feat_dims + [feat_dims[-1]]
        cur_up_feat_dims = cur_up_feat_dims[1:]
        prv_up_feat_dims = self.feat_dims[:-1]
        prv_up_feat_dims[0] += 3
        out_up_feat_dims = up_feat_dims
        in_up_feat_dims = [cur_up + prv_up for cur_up, prv_up in zip(cur_up_feat_dims, prv_up_feat_dims)]

        self.pointnetpp_conv_layers = nn.ModuleList()
        self.edgeconv_layers = nn.ModuleList()
        self.transition_up_layers = nn.ModuleList()
        self.rsconv_layers = nn.ModuleList()
        self.local_conv_with_glb_layers = nn.ModuleList()
        # self.pure_mlp_layers = nn.ModuleList()

        self.radius = radius

        self.in_dims = self.feat_dims[:-1]
        self.out_dims = self.feat_dims[1:]

        for i, (in_dim, out_dim) in enumerate(zip(self.in_dims, self.out_dims)):
            self.pointnetpp_conv_layers.append(
                TransitionDown(
                    feat_dim=in_dim + 3,
                    out_feat_dim=out_dim, k=self.k, last_act=False, conv_type="pointnetpp", abstract=True
                )
            )
            self.edgeconv_layers.append(
                TransitionDown(
                    feat_dim=in_dim + 3,
                    out_feat_dim=out_dim, k=self.k, last_act=False, conv_type="edgeconv", abstract=False
                )
            )

            self.local_conv_with_glb_layers.append(
                TransitionDown(
                    feat_dim=in_dim + 3,
                    out_feat_dim=out_dim, k=self.k, last_act=False, conv_type="localconv", abstract=False
                )
            )


        # construct TransitionUp modules
        up_conv_final_dim = self.feat_dims[0] if self.feat_dims[0] > 0 else self.feat_dims[1]
        for cur_up_in, cur_up_out in zip(reversed(in_up_feat_dims), reversed(out_up_feat_dims)):
            self.transition_up_layers.append(
                TransitionUp(fea_in=cur_up_in, fea_out=cur_up_out)
            )

        map_feat_out_dim = up_conv_final_dim # self.feat_dims[0]
        map_feat_out_dim = out_up_feat_dims[0]

        self.feat_map_layer_out = construct_conv1d_modules(
            [map_feat_dim, map_feat_dim], n_in=map_feat_out_dim, last_act=False
        )

    def forward(self, x: torch.FloatTensor, pos: torch.FloatTensor, return_global=False, use_ori_feat=True, r=None,
                conv_select_types=[0, 0, 0, 0, 0], relative_pos=False, edgeconv_interpolate=True, edgeconv_skip_con=True,
                radius=None
                ):
        # bz, N = x.size(0), x.size(1)
        # # only for feature map; no issue w.r.t. the communication between different points' features
        # ###### For relative feature test...

        # x = x[:, :, 3:]

        cache = list()
        cache.append((pos.clone(), x.clone() if x is not None else None))

        # ###### GET
        down_modules = []
        up_modules = []
        act_layers = self.n_layers

        # conv_select_sampled_rates = torch.rand((act_layers, ), dtype=torch.float32)

        i_layers = 0
        conv_types = []
        for i in range(act_layers):
            if conv_select_types[i] == 0:
                down_modules.append(self.pointnetpp_conv_layers[i_layers])
                up_modules = [self.transition_up_layers[self.n_layers - 1 - i_layers]] + up_modules
                i_layers += 1
                conv_types.append("pointnetpp")
            elif conv_select_types[i] == 1:
                down_modules.append(self.edgeconv_layers[i_layers])
                up_modules = [self.transition_up_layers[self.n_layers - 1 - i_layers]] + up_modules
                i_layers += 1
                conv_types.append("edgeconv")
            elif conv_select_types[i] == 4:
                down_modules.append(self.rsconv_layers[i_layers])
                up_modules = [self.transition_up_layers[self.n_layers - 1 - i_layers]] + up_modules
                i_layers += 1
                conv_types.append("rsconv")
            elif conv_select_types[i] == 3:
                down_modules.append(self.local_conv_with_glb_layers[i_layers])
                up_modules = [self.transition_up_layers[self.n_layers - 1 - i_layers]] + up_modules
                i_layers += 1
                conv_types.append("localconv")
            elif conv_select_types[i] == 2:
                continue
        # x: torch.FloatTensor, pos: torch.FloatTensor, n_sampling: int=None, relative_pos=False,
        #                 return_fps=False, r=None, similarity_calculation_feat=None
        i_down_sample = 0
        for i, layer in enumerate(down_modules):
            if conv_types[i] == "pointnetpp":
                x, pos = layer(x, pos, self.n_samples[i_down_sample], relative_pos=relative_pos,
                               r=None if radius is None else radius[i]
                               )
                cache.append((pos.clone(), x.clone()))
                i_down_sample += 1
            elif conv_types[i] == "edgeconv":
                # no down-sampling is applied
                x, pos = layer(
                    x, pos, self.n_samples[i_down_sample], relative_pos=relative_pos, r=r,
                    similarity_calculation_feat=pos if i == 0 else x
                )
                # print(f"After {i}-th edgeconv layer, x.size = {x.size()}, pos.size = {pos.size}")
                cache.append((pos.clone(), x.clone()))
            elif conv_types[i] == "localconv":
                x, pos = layer(
                    x, pos, self.n_samples[i_down_sample], relative_pos=relative_pos, r=r,
                )
                cache.append((pos.clone(), x.clone()))
            elif conv_types[i] == "rsconv":
                x, pos = layer(
                    x, pos, self.n_samples[i_down_sample], relative_pos=relative_pos, r=r,
                )
                cache.append((pos.clone(), x.clone()))
                i_down_sample += 1 # with down-sampling

        up_x = x
        # print(up_x.size())
        for i, layer in enumerate(up_modules):
            if conv_types[i_layers - 1 - i] in ["pointnetpp", "rsconv"]:
                prv_feat = cache[-i - 2][1] if use_ori_feat else None
                if prv_feat is not None and i == len(up_modules) - 1:
                    prv_feat = torch.cat(
                        [prv_feat, cache[-i - 2][0]], dim=-1
                    )
                elif i == len(up_modules) - 1:
                    prv_feat = cache[-i - 2][0]
                up_x, pos = layer(up_x, pos, prv_feat, cache[-i - 2][0])
            elif conv_types[i_layers - 1 - i] == "edgeconv":
                up_x, pos = layer(
                    up_x, cache[-i - 2][0], cache[-i - 2][1] if use_ori_feat else None, cache[-i - 2][0],
                    interpolate=edgeconv_interpolate, skip_connection=edgeconv_skip_con
                )
            elif conv_types[i_layers - 1 - i] == "localconv":
                up_x, pos = layer(
                    up_x, pos, cache[-i - 2][1] if use_ori_feat else None, cache[-i - 2][0],
                    interpolate=False, skip_connection=False
                )

        # up_x = self.apply_module_with_bn(up_x, self.feat_map_layer_out)
        up_x = CorrFlowPredNet.apply_module_with_conv1d_bn(
            up_x, self.feat_map_layer_out
        )
        if return_global:
            return up_x, x, pos
        else:
            return up_x, pos
#
