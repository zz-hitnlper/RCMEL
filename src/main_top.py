from omegaconf import OmegaConf
import os,sys
from embedding import run_emb,run_top_multigpu,run_top
from maketop50 import maketop50
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
def setup_parser(config_path):
    return OmegaConf.load(config_path)

if __name__ == "__main__":
    # dataset="wikidiverse"
    dataset = sys.argv[1]
    config_path = f"/home/xxx/code/mel/rcmel-my/config/{dataset}.yaml"
    args = setup_parser(config_path)
    # run_emb(args)
    # maketop50(dataset)
    run_top_multigpu(args)
    print("===")
    # run_top(args)