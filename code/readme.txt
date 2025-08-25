下游任务：
下游任务的模型参数配置在Downstream_model下的target_encoder配置里，主要是embed_dim、depth的调整，这里只需要配置encoder来微调
/home/zehaoqin/ST-MEM/code 这一套代码对应的ckpt在/ssd1/qinzehao/new_v2/v3/pretrain下
运行指令是bash run_downstream.sh --config_path=/home/zehaoqin/ST-MEM/code/configs/downstream/st_mem_ptb.yaml


预训练：
代码比较乱有多个版本，encoder、predictor、decoder部分都在多个new_model文件里，比如/home/zehaoqin/ST-MEM/new_model_v4.py
这里我就没整理出来了，应该也不需要跑预训练代码，如果想跑，要先cd到/home/zehaoqin/ST-MEM这个路径下
然后运行bash run_pretrain.sh --config_path=/home/zehaoqin/ST-MEM/configs/pretrain/st_mem.yaml 记得换一下output_dir的路径，避免把我的覆盖掉

我觉得主要还是看模型的设计就可以，如果和文本结合，直接拿现在的预训练权重也不合适，还是要拿心电编码器和文本编码器重新联合训练，或者拿心电编码器训好的embedding送进LLM