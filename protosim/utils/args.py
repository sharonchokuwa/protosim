import argparse
from configs.defaults import _C as cfg_default

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/datasets", help="path to dataset")
    parser.add_argument("--algorithm", type=str, default='ProtoSim', help='check in algorithms.py')
    parser.add_argument("--desc", type=str, default="name_of_exp")
    parser.add_argument("--backbone", type=str, default="resnet50")
    #parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DGDR")
    #parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DGDR")
    #parser.add_argument("--random", action="store_true") 
    parser.add_argument("--dg_mode", type=str, default='DG', help="DG or ESDG")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ep", type=int, default=5)
    parser.add_argument("--output", type=str, default='test') 
    parser.add_argument("--override", action="store_true") 

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--center_loss_lr", type=float, default=0.001)
    parser.add_argument("--sim_loss_lr", type=float, default=0.5)

    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--sim_alpha", type=float, default=1000)
    parser.add_argument("--center_alpha", type=float, default=0.001)
    parser.add_argument("--ce_alpha", type=float, default=1)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)

    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--weight_decay_other", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optim", type=str, default='sgd')
    parser.add_argument("--proto_optim", type=str, default='sgd')
    parser.add_argument("--timestamp", type=str, default='not_resuming', help="useful for resuming training")
    parser.add_argument("--project_name", type=str, default='Similarity', help="Main project name")
    parser.add_argument("--focalloss", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--target", type=str, default='deepdr')
    parser.add_argument("--save_prototypes", action="store_true")
    
    return parser.parse_args()
#
def setup_cfg(args):
    cfg = cfg_default.clone()
    #cfg.RANDOM = args.random
    #cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    #cfg.DATASET.TARGET_DOMAINS = args.target_domains
    cfg.SEED = args.seed
    cfg.OUTPUT_PATH = args.output
    cfg.OVERRIDE = args.override
    cfg.DG_MODE = args.dg_mode
    cfg.DESCRIPTION = args.desc
    cfg.ALGORITHM = args.algorithm
    cfg.BACKBONE = args.backbone

    cfg.SIGMA = args.sigma
    cfg.SIM_ALPHA = args.sim_alpha
    cfg.CENTER_ALPHA = args.center_alpha
    cfg.CE_ALPHA = args.ce_alpha
    
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_CLASSES = args.num_classes

    cfg.VAL_EPOCH = args.val_ep
    cfg.EPOCHS = args.num_epochs

    cfg.OPTIMIZER = args.optim
    cfg.PROT_OPTIMIZER = args.proto_optim
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.MOMENTUM = args.momentum
    cfg.WEIGHT_DECAY_OTHER = args.weight_decay_other

    cfg.LEARNING_RATE = args.lr
    cfg.CENTER_LOSS_LR = args.center_loss_lr
    cfg.SIM_LOSS_LR = args.sim_loss_lr

    cfg.BATCH_SIZE = args.batch_size
    cfg.VAL_BATCH_SIZE = args.val_batch_size

    cfg.TIMESTAMP = args.timestamp
    cfg.PROJECT_NAME = args.project_name
    cfg.TARGET = args.target
    cfg.DEBUG = args.debug
    cfg.FOCALLOSS = args.focalloss
    cfg.SAVE_PROTOTYPES = args.save_prototypes
    
    if args.dg_mode == 'DG':
        cfg.merge_from_file("./configs/datasets/GDRBench.yaml")
    elif args.dg_mode == 'ESDG':
        cfg.merge_from_file("./configs/datasets/GDRBench_ESDG.yaml")
    else:
        raise ValueError('Wrong type')

    return cfg

