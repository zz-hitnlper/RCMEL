from omegaconf import OmegaConf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
import sys
# 实体增强
from entity_aug import entity_aug_multigpu_mllm, entity_aug_multigpu_llm,chaquebuloumllm


def setup_parser(config_path):
    return OmegaConf.load(config_path)

if __name__ == "__main__":
    # dataset="wikidiverse"
    dataset = sys.argv[1]
    config_path = f"/home/xxx/code/mel/rcmel-my/config/{dataset}.yaml"
    args = setup_parser(config_path)
    # 实体增强
    # 单gpu
    # entity_aug_onegpu_mllm(args)
    # chaquebuloumllm(args)
    # entity_aug_onegpu_llm(args)
    # 多gpu
    # entity_aug_multigpu_mllm(args)
    # chaquebuloumllm(args)
    entity_aug_multigpu_llm(args)
