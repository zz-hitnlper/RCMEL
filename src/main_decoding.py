import argparse
from omegaconf import OmegaConf
import os,sys
from decoding import infer_decoding_multigpu
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
def setup_parser(dataset):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=f"/home/xxx/code/mel/rcmel-my/config/{dataset}.yaml", type=str)
    _args = parser.parse_args()
    args = OmegaConf.load(_args.config)
    return args

if __name__ == "__main__":
    dataset = sys.argv[1]
    config_path = f"/home/xxx/code/mel/rcmel-my/config/{dataset}.yaml"
    args = setup_parser(dataset)
    infer_decoding_multigpu(args)