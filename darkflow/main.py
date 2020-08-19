from darkflow.utils.attr_utils import AttributeDict
import json
import argparse


def parse():
    parser = argparse.ArgumentParser(description="darkflow_configs")
    parser.add_argument('--train_net', default=False, type=bool)
    parser.add_argument('--test_net', default=True, type=bool)
    parser.add_argument('--configs_file', type=str, default='/home/pjawahar/Projects/DarkFlow/darkflow/configs/configs.json')
    parser.add_argument('--network', type=str, default='convnet', choices=['convnet'])
    parser.add_argument('--flow', type=str, default='none', choices=['none', 'planar', 'orthosnf', 'householdersnf', 'triangularsnf', 'iaf', 'convflow'])
    args = parser.parse_args()
    return args


def run(args):
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