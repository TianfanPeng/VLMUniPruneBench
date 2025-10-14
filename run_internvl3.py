from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.dynamic_module_utils import get_class_from_dynamic_module
import os

# 自定义 generate 函数（从 custom_internvl.py 导入）
# InternVL3Chat_generate 处理 pre-LLM 剪枝
# Qwen2Model_forward 处理 intra-LLM 剪枝
from custom_internvl import InternVLChatModel_generate
from custom_internvl import Qwen2Model_forward


if __name__ == '__main__':

    chat_cls = get_class_from_dynamic_module("modeling_internvl_chat.InternVLChatModel", os.getenv("INTERNVL3_MODEL_PATH", "OpenGVLab/InternVL3-1B"))
    chat_cls.generate = InternVLChatModel_generate

    Qwen2Model.forward = Qwen2Model_forward
    
    # 运行 VLMEvalKit 主函数
    from VLMEvalKit.run import main as run_main
    run_main()
