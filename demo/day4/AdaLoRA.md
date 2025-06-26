# LoRA应用策略
偷懒看了一下小米发布会，明天开始租算力训练模型，先从单卡开始
Transformer Attention 中的：
* q_proj（query projection）
* k_proj（key projection）
* v_proj（value projection）
* out_proj（output projection）
Feed-Forward Network（MLP）中的：
* fc1（第一层线性映射）
* fc2（第二层线性映射）
前几天对于LoRA的学习太肤浅了，看了Parameter-Efficient Fine-Tuning with Adaptively Learned Low-Rank Adaptation之后，发现AdaLoRA可以让模型自己学习每个 LoRA 层该用多少秩，相同 FLOPs，AdaLoRA > LoRA > Full Fine-tuning
* get_peft_model() 会自动在指定的层插入 LoRA 层。
* 使用 prepare_model_for_kbit_training() 兼容 8bit/fp16 训练场景，冻结其他层、启用梯度。
`AdaLoraConfig(
    init_r=12,          # 初始秩（刚开始所有模块都是 r=12）
    target_r=4,         # 最终目标秩（训练结束时，压缩到 r=4）
    beta1=0.85, beta2=0.85,  # 类似 Adam 优化器的动量参数，控制秩压缩过程的平滑程度
    tinit=200,          # 第 200 步之后开始压秩
    tfinal=1000,        # 第 1000 步达到最终目标秩
    deltaT=10,          # 每 10 步更新一次秩
    lora_alpha=32,      # 缩放因子，跟普通 LoRA 一样
    lora_dropout=0.1,   # 防止过拟合
    orth_reg_weight=0.5,# 增加一个正交正则项，约束 A、B 更稳定地收敛
    target_modules=..., # 插入的位置（注意力层、MLP）
    total_step=total_step # 总训练步数，帮助计算 tinit 和 tfinal
)
`
