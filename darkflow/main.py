from darkflow.utils.attr_utils import AttributeDict
import json
import argparse


def parse():
    parser = argparse.ArgumentParser(description="darkflow_configs")
    parser.add_argument('--mode', type=str, default='testing', choices=['training', 'testing'], help='Choose if are you training or testing the model')
    parser.add_argument('--configs_file', type=str, default='/home/pjawahar/Projects/DarkFlow/darkflow/configs/configs.json')
    parser.add_argument('--network', type=str, default='convnet', choices=['convnet'])
    parser.add_argument('--flow', type=str, default='noflow', choices=['noflow', 'planar', 'orthosnf', 'householdersnf', 'triangularsnf', 'iaf', 'convflow'])
    args = parser.parse_args()
    return args


def run(args):
    if args.mode == 'training':
        print('Preparing to Train Model')
        args.train_net = 1
    elif args.mode == 'testing':
        print('Preparing to Test Model')
        args.test_net = 1
    else:
        raise argparse.ArgumentTypeError('Run mode not chosen correctly. Choose if you want to train (--mode training) or test (--mode testing).')

    if args.network == 'convnet':
        from darkflow.runners.convnet_runner import ConvNetRunner
        network = ConvNetRunner(args=args)

        if args.train_net:
            network.trainer()
            
        if args.test_net:
            network.tester()
    

if __name__ == '__main__':
    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    configs = {**configs, **args}
    configs = AttributeDict(configs)
    run(configs)

"""
Current Configs (Chan3):

 "num_classes": 1,
 "training_fraction": 0.98,
 "batch_size": 16,
 "test_batch_size": 1,
 "learning_rate": 0.001,
 "latent_dim": 15,
 "beta": 1,       
 "num_epochs": 10,

 "q_z_output_dim" : 20,
 "num_flows" : 6,         # 4 gave problematic ROC for PF; Keep at 4 for ConvF(0.92AUC) and the rest
 "num_ortho_vecs" : 8,
 "num_householder" : 12,  # 8 gave problematic ROC
 "made_h_size" : 330,
 "convFlow_kernel_size" : 7  # 5 gave problematic ROC


Current Configs (Chan1):
 "num_classes": 1,
 "training_fraction": 0.98,
 "batch_size": 16,
 "test_batch_size": 1,
 "learning_rate": 0.0001,
 "latent_dim": 15,
 "beta": 1,       
 "num_epochs": 10,

 "q_z_output_dim" : 20,
 "num_flows" : 3,         # 4 for iaf, tri, house; 6 for convF
 "num_ortho_vecs" : 5,   
 "num_householder" : 8,  # 6
 "made_h_size" : 330,
 "convFlow_kernel_size" : 7 


Current Configs (Chan2b):
 "num_classes": 1,
 "training_fraction": 0.98,
 "batch_size": 16,
 "test_batch_size": 1,
 "learning_rate": 0.0001,
 "latent_dim": 15,
 "beta": 1,       
 "num_epochs": 10,

 "q_z_output_dim" : 20,
 "num_flows" : 3,         # 4 for iaf, tri, house; 6 for convF
 "num_ortho_vecs" : 5,   
 "num_householder" : 8,  
 "made_h_size" : 330,
 "convFlow_kernel_size" : 7  
"""