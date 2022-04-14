# from trainer import TrainerClassification
import argparse
import torch
import numpy as np
import os
import logging
import sys
import random
import munch
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('--local_rank', type=int, default=0, help='whether switch to debug module')
    # print("here 3!")
    args = parser.parse_args()
    tmp_local_rank = args.local_rank
    # args.config = "./cfgs/prim_dist.yaml"

    config_path = args.config
    # print("here 4!")
    args = munch.munchify(yaml.safe_load(open(config_path)))
    args.local_rank = tmp_local_rank

    if args.task == "sea_multi_loss_h_v2_tree_optim_loss":
        from trainer.prim_trainer_sea_h_v2_multi_loss_v2_tree_train_losses import TrainerPrimitiveFitting as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "sea_multi_loss_h_mb_v2_tree_optim_loss":
        from trainer.prim_trainer_sea_h_v2_multi_loss_mb_v2_tree_train_losses import TrainerPrimitiveFitting as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "sea_multi_loss_h_mb_v2_tree":
        from trainer.prim_trainer_sea_h_v2_multi_loss_mb_v2_tree import TrainerPrimitiveFitting as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "sea_multi_loss_h_ob_v2_tree":
        from trainer.prim_trainer_sea_h_v2_cross_multi_loss_v2_tree import TrainerPrimitiveFitting as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "prim_inference":
        from trainer.prim_inference import TrainerPrimitiveFitting as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "inst_seg_inference":
        from trainer.instseg_inference import TrainerInstSegmentation as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "inst_seg_sea_h_ob_cross_v2_partnet":
        from trainer.instseg_trainer_sea_h_ob_cross_multi_loss_v2_partnet import TrainerInstSegmentation as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "inst_seg_sea_h_mb_v2_partnet":
        from trainer.instseg_trainer_sea_h_mb_multi_loss_v2_partnet import TrainerInstSegmentation as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "motion_seg_sea_h_ob_cross":
        from trainer.motionseg_trainer_sea_h_ob_cross_multi_loss import TrainerInstSegmentation as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "motion_seg_sea_h_mb_cross":
        from trainer.motionseg_trainer_sea_h_mb_cross_multi_loss import TrainerInstSegmentation as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    elif args.task == "motion_inference":
        from trainer.motionseg_inference import TrainerInstSegmentation as Trainer
        dataset_root = "/data-input/motion_part_split_meta_info"
    else:
        raise ValueError("Unrecognized task name. Expected cls or sem_seg, got %s" % args.task)

    cudaa = args.device if torch.cuda.is_available() else None
    trainer = Trainer(dataset_root=dataset_root,
                      num_points=args.num_points,
                      batch_size=args.batch_size,
                      num_epochs=args.epochs,
                      cuda=cudaa,
                      use_sgd=args.use_sgd,
                      weight_decay_sgd=args.weight_decay_sgd,
                      resume=args.resume,
                      dp_ratio=args.dp_ratio,
                      args=args
                      )

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(trainer.model_dir, 'logs.txt')),
                                                      logging.StreamHandler(sys.stdout)])
    logging.info(str(args))
    logging.info("Start training...")
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logging.info("Random seed = %d", seed)
    logging.info("Local rank = %d", args.local_rank)
    trainer.train_all()
