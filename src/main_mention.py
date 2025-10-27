from omegaconf import OmegaConf
import os,sys
# 实体增强
from mention_aug import mention_aug_multigpu_mllm, mention_aug_multigpu_llm,mention_aug_multigpu,mention_aug_onegpu
# 设置使用的显卡序号（例如使用第 0 号显卡）
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
def setup_parser(config_path):
    return OmegaConf.load(config_path)

if __name__ == "__main__":
    # dataset="wikidiverse"
    dataset = sys.argv[1]
    config_path = f"/home/xxx/code/mel/rcmel-my/config/{dataset}.yaml"
    args = setup_parser(config_path)
    # 提及增强
    # mention_aug_onegpu(args)# 单gpu
    # # mention_aug_multigpu(args)# 多gpu
    mention_aug_multigpu_mllm(args)
    mention_aug_multigpu_llm(args)
    # 计算并保存检索
    
    # 获取top-K
    # LLM问答