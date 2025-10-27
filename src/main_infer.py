from omegaconf import OmegaConf
import os,sys
from infer import infer_kc_multigpu,infer_kc_onegpu
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
def setup_parser(config_path):
    return OmegaConf.load(config_path)

if __name__ == "__main__":
    # dataset="RichpediaMEL"
    dataset = sys.argv[1]
    config_path = f"/home/xxx/code/mel/rcmel-my/config/{dataset}.yaml"
    args = setup_parser(config_path)
    infer_kc_multigpu(args)
    # infer_kc_onegpu(args)