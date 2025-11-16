from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel, Qwen2_5_VLTextModel


from custom_qwenvl import Qwen2_5_VLModel_forward, Qwen2_5_VLTextModel_forward



if __name__ == "__main__":
    Qwen2_5_VLTextModel.forward = Qwen2_5_VLTextModel_forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
    
    # 运行 VLMEvalKit 主函数
    from VLMEvalKit.run import main as run_main
    run_main()