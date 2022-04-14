

from trainer.trainer_utils import DistributionTreeNodeV2, DistributionTreeNodeArch
from model.constants import *
import munch
import yaml
import argparse
import numpy as np

nn_grp_opers = GROUP_INVALID
nn_binary_opers =  6
nn_unary_opers =  6
nn_in_feats = 7

parser = argparse.ArgumentParser(description='Train config file')
parser.add_argument('-c', '--config', help='path to config file', required=True)
parser.add_argument('--ty', type=str, default="loss", help='whether switch to debug module')
parser.add_argument('--local_rank', type=int, default=0, help='whether switch to debug module')
args = parser.parse_args()
tmp_local_rank = args.local_rank
ty = args.ty

config_path = args.config
args = munch.munchify(yaml.safe_load(open(config_path)))
args.local_rank = tmp_local_rank

sv_params_path = ""

try:
    preload_params = np.load(sv_params_path, allow_pickle=True).item()
    print(type(preload_params))
except:
    preload_params = None
    pass

K = 50

if ty == "loss":

    print(preload_params.keys())
    print(preload_params["uop"])
    print(type(preload_params["uop"]))
    print(f"less_layers = {args.less_layers}")
    sampling_tree_rt = DistributionTreeNodeV2(cur_depth=0 if not args.less_layers else 1, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feats, args=args, preload_params=preload_params)
elif ty == "loss_rnd":
    sampling_tree_rt = DistributionTreeNodeV2(cur_depth=0, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                              nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feats, args=args,)
else:
    sampling_tree_rt = DistributionTreeNodeArch(cur_depth=0, tot_layers=3, nn_conv_modules=4, device=None,
                                                      args=None, preload_params=preload_params)

sampled_res = []
for j in range(K):
    cur_res = sampling_tree_rt.sampling(cur_depth=0 if not args.less_layers else 1)
    if ty == "arch":
        possi = sampling_tree_rt.calcu_possi(cur_res, 0 if not args.less_layers else 1)
        sampled_res.append((cur_res, possi))
    else:
        possi = sampling_tree_rt.calcu_possi(cur_res, 0 if not args.less_layers else 1)
        sampled_res.append((cur_res, possi))

sampled_res = sorted(sampled_res, key=lambda ii: ii[1], reverse=True)
print(sampled_res)

