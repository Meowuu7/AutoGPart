### AutoGPart: Intermediate Supervision Search for Generalizable 3D Part Segmentation

AutoGPart is a method that builds an intermediate supervision space to search from to improve the generalization ability of 3D part segmentation networks. 

This repository contains PyTorch implementation of our paper: 

[AutoGPart: Intermediate Supervision Search for Generalizable 3D Part Segmentation](https://arxiv.org/pdf/2203.06558.pdf), *Xueyi Liu*, *Xiaomeng Xu*, [*Anyi Rao*](https://anyirao.com), [*Chuang Gan*](https://people.csail.mit.edu/ganchuang/), [*Li Yi*](https://ericyi.github.io), CVPR 2022.

![Screen Shot 2022-04-14 at 5.46.04 PM](./assets/overall-pipeline-23-1.png)

## Links

- [Project Page](https://autogpart.github.io) (including videos, visualizations for searched intermediate supervisions and segmentations)
- [arXiv Page](https://arxiv.org/abs/2203.06558)

## Environment and package dependency

The main experiments are implemented on PyTorch 1.9.1, Python 3.8.8. Main dependency packages are listed as follows:

```
torch_cluster==1.5.9
torch_scatter==2.0.7
horovod==0.23.0
pykdtree==1.3.4
numpy==1.20.1
h5py==2.8.0
```

## Supervision search stage

### Mobility-based part segmentation

To create and optimize the intermediate supervision space for the mobility-based part segmentation task, please use the following command (single machine):

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/motion_seg_h_mb_cross.yaml
```

The default backbone is DGCNN. 

### Primitive fitting

To create and optimize the supervision distribution space for the primitive fitting task, please use the following command:

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/prim_seg_h_mb_cross_v2_tree.yaml
```

The default backbone is DGCNN.

To optimize the supervision distribution space for the first stage of primitive fitting task using HPNet-style network architecture, please use the following command:

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/prim_seg_h_mb_v2_tree_optim_loss.yaml
```

### Sample supervision features from the optimized supervision feature distribution space

The following command samples a set of operations with relatively high sampling probabilities from the optimized supervision space (distribution parameters are stored in `dist_params.npy` under the logging directory):

```shell
python load_and_sample.py -c cfgs/${your_config_file} --ty=loss --params-path=${your_parameter_file}
```

### Greedy search for optimal supervison combinations

After optimizing the supervision feature space, we should sample proper supervision features for further use. There are two strategies: 

- Pick features with top sampling probabilities from the optimization distributions.
- Sample a certain number of supervision features from the optimized space (*e.g.* 10), and then use the greedy search process to choose a combination of features from them. 

For the second strategy, the parameter `beam_search` in the config file should be set to `True`. Then use the following commands three times to select a combination of supervision features: 

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/${your_config_file}
```

The function `beam_searh_for_best` in each trainer file should be properly modified to guide the selection process according to the algorithm described in the supplementary material.

## Training stage

To train a segmentation network with selected supervision features, the corresponding trainer file should be modified by plugging the selected features into proper code lines. then for each segmentation task: 

### Mobility-based part segmentation

Use the following command to train the network together with selected supervision features: 

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/motion_seg_h_ob_cross_tst_perf.yaml
```

### Primitive fitting

Use the following command to train the network together with selected supervision features: 

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/prim_seg_h_ob_v2_tree.yaml
```

## Inference stage

### Mobility-based part segmentation

Replace `resume` in motion_seg_inference to the path to saved model weights and use the following command to evaluate the trained model:

```shell
python -W ignore main_prm.py -c ./cfgs/motion_seg_inference.yaml
```

Remember to select a free GPU in the config file.

### Primitive fitting

Replace `resume` in `prim_inference.yaml` to the path to saved model weights and use the following command to evaluate the trained model:

```shell
python -W ignore main_prm.py -c ./cfgs/prim_inference.yaml
```

Remember to select a free GPU in the config file.

You should modify the file `prim_inference.py` to choose whether to use the clustering-based segmentation module or classification-based one.

For clustering-based segmentation, use the `_clustering_test` function; For another, use the `_test` function.

## Datasets

(TODO: post-processed data)

### Mobility-based part segmentation

We collect the training dataset and the auxiliary training dataset from [1,2] for the mobility-based part segmentation task. We infer mobility meta-data heuristically for parts in each shape. Original datasets can be downloaded from [ShapeNetPart](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip) and [PartNet](https://www.shapenet.or).

The test dataset is the same as the one used in [3] (which could be downloaded via [PartMob](https://shapenet.cs.stanford.edu/ericyi/pretrained_model_partmob.zip)). We use the trained flow estimation model to estimate flow for each shape and save the data with coordinate information for spare time. During the inference, we directly use the estimated flow.

### Primitive fitting

We use the same dataset as the one used in [4]. We re-split the dataset to better test the domain generalization ability of the model.

The dataset can be downloaded via [Traceparts](https://www.traceparts.com/) (original version). 

## Data

### Mobility-based part segmentation

## Checkpoints

There are two kinds of checkpoints: 

- Optimized distributions for the constructed supervision feature space
- Pre-trained segmentation network using the searched intermediate supervisions

### Mobility-based part segmentation

Please download optimized distribution parameters and trained models from [here](https://drive.google.com/drive/folders/1oPocnUABlkRbO9wmwmKHCy2VM-BZrUDm?usp=sharing).

## Unsolved Problems

- ***Unfriendly operations***: Sometimes the model will sample operations which would result in a supervision feature with very large absolute values. It would scarcely hinder the optimization process (since such supervisions would cause low metric values; thus, the model using them will not be passed to the next step), making the optimization process ugly. 

  The problem could probably be solved by forbidding certain operation combinations/sequences. And please feel free to submit a pull request if you can solve it. 

- ***Unnormalized rewards***: Reward values used for such three tasks may have different scales. It may affect the optimization process to some extent. They could probably be normalized using prior knowledge of generalization gaps of each task and its corresponding training data. 

## Reference

Part of the code is taken from [HPNet](https://github.com/SimingYan/HPNet), [SPFN](https://github.com/lingxiaoli94/SPFN), [Deep Part Induction](https://github.com/ericyi/articulated-part-induction), [PointNet2](https://github.com/charlesq34/pointnet2), [MixStyle](https://github.com/KaiyangZhou/mixstyle-release).

## License

Our code and data are released under MIT License (see LICENSE file for details).



[1] Yi, L., Kim, V. G., Ceylan, D., Shen, I. C., Yan, M., Su, H., ... & Guibas, L. (2016). A scalable active framework for region annotation in 3d shape collections. *ACM Transactions on Graphics (ToG)*, *35*(6), 1-12.

[2] Mo, K., Zhu, S., Chang, A. X., Yi, L., Tripathi, S., Guibas, L. J., & Su, H. (2019). Partnet: A large-scale benchmark for fine-grained and hierarchical part-level 3d object understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 909-918).

[3] Yi, L., Huang, H., Liu, D., Kalogerakis, E., Su, H., & Guibas, L. (2018). Deep part induction from articulated object pairs. *arXiv preprint arXiv:1809.07417*.

[4] Li, L., Sung, M., Dubrovina, A., Yi, L., & Guibas, L. J. (2019). Supervised fitting of geometric primitives to 3d point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2652-2660).

