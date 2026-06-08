import os
import torch
import numpy as np
import random
from tqdm import tqdm
import time
import torch.nn.functional as F

# --- 1. 导入你的自定义模块 ---
# 请确认你的模型文件名是 model_MC_FT.py 还是 model_MC_AM.py，这里我先按你之前发的写
from model.model_MC_FT import VQAModel 
from dataloader.VQALoader import VQALoader
from transformers import RobertaModel, RobertaTokenizerFast, AutoImageProcessor

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 根据实际情况修改
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 2. 种子固定函数 ---
def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- 3. 评估函数 (完全仿照参考代码) ---
def evaluate(model, test_loader, device, checkpoint_path):
    print(f"\n--- 开始在测试集上进行最终评估 ---")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 找不到模型权重文件 '{checkpoint_path}'")
        return

    print(f"正在从 '{checkpoint_path}' 加载模型权重...")
    
    # === 关键修改：处理 Lightning 的 ckpt 文件结构 ===
    # 因为你的权重是 PyTorch Lightning 保存的，它把权重字典放在了 "state_dict" 这个 key 下面
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 加载权重 (strict=False 允许忽略一些无关的层，比如 loss 函数参数)
    model.load_state_dict(state_dict, strict=False)
    print("模型权重加载成功！✅")

    model.to(device)
    model.eval()

    # 初始化计数器
    total_rural_urban, total_presence, total_count, total_comp = 0, 0, 0, 0
    right_rural_urban, right_presence, right_count, right_comp = 0, 0, 0, 0
    test_step_outputs = []

    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, total=len(test_loader), desc="🧪 Testing")
        for data in test_progress_bar:
            (
                pixel_values, input_ids, token_type_ids, attention_mask, answer, 
                question_type, img_id, question, answer_str
            ) = data
            
            # 数据移至 GPU
            pixel_values = pixel_values.float().to(device)
            input_ids = input_ids.long().to(device)
            token_type_ids = token_type_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            # 注意：answer 在这里主要用于计算准确率，转为 numpy
            answer_test = answer.long().to(device).cpu().numpy()

            # 前向传播
            # 注意：这里根据你的 VQAModel forward 参数进行传递
            pred = model(pixel_values, input_ids, token_type_ids, attention_mask)
            
            # 获取预测结果
            pred_arg = np.argmax(pred.cpu().detach().numpy(), axis=1)

            # 统计
            for j in range(pred_arg.shape[0]):
                if pred_arg[j] == answer_test[j]:
                    test_step_outputs.append([1, question_type[j]])
                else:
                    test_step_outputs.append([0, question_type[j]])

    print("\n--- 测试集评估结果 ---")
    outputs = np.array(test_step_outputs)
    
    # 你的参考代码里的 categories 是写死的判断逻辑，这里保持一致
    for i in range(outputs.shape[0]):
        q_type = outputs[i, 1]
        is_correct = int(outputs[i, 0]) # '1' -> 1
        
        # 请根据你的数据集实际 type 字符串修改这里
        # 比如有的数据集是 "Comparison" 而不是 "comp"
        if q_type == "comp" or q_type == "Comparison":
            total_comp += 1
            right_comp += is_correct
        elif q_type == "presence" or q_type == "Presence":
            total_presence += 1
            right_presence += is_correct
        elif q_type == "count" or q_type == "Count":
            total_count += 1
            right_count += is_correct
        else: # 默认为 area / rural_urban
            total_rural_urban += 1
            right_rural_urban += is_correct
            
    # 计算准确率
    acc_rural_urban = (right_rural_urban / total_rural_urban * 100) if total_rural_urban > 0 else 0
    acc_presence = (right_presence / total_presence * 100) if total_presence > 0 else 0
    acc_count = (right_count / total_count * 100) if total_count > 0 else 0
    acc_comp = (right_comp / total_comp * 100) if total_comp > 0 else 0

    right = right_rural_urban + right_presence + right_count + right_comp
    total = total_rural_urban + total_presence + total_count + total_comp

    AA = (acc_rural_urban + acc_presence + acc_count + acc_comp) / 4
    OA = (right / total * 100) if total > 0 else 0
    
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f'[{current_time}] 📊 最终评估指标:')
    print(f'➡️  Acc_area (rural/urban): {acc_rural_urban:.2f}% ({right_rural_urban}/{total_rural_urban})')
    print(f'➡️  Acc_presence: {acc_presence:.2f}% ({right_presence}/{total_presence})')
    print(f'➡️  Acc_count: {acc_count:.2f}% ({right_count}/{total_count})')
    print(f'➡️  Acc_comp: {acc_comp:.2f}% ({right_comp}/{total_comp})')
    print(f'---------------------------------')
    print(f'⭐  Average Accuracy (AA): {AA:.2f}%')
    print(f'🎯  Overall Accuracy (OA): {OA:.2f}% ({right}/{total})')
    print("评估完成！🚀")


# --- 4. 主程序 ---
if __name__ == "__main__":
    print("--- VQA 模型评估脚本 (Native PyTorch Style) ---")

    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # === 参数设置 ===
    batch_size = 32
    sequence_length = 40
    num_workers = 4
    NUM_CLASS = 94
    
    data_path = "RSVQA_HR"
    # 请修改为你的 ckpt 路径
    checkpoint_path = "checkpoint1/epoch=10_valid_AA=0.85354.ckpt" 

    seed_torch(seed)
    print(f"使用的设备: {device}")
    print(f"模型文件: {checkpoint_path}")

    # === 初始化处理器 (仿照参考代码) ===
    print("初始化 Tokenizer 和 Image Processor...")
    tokenizer = RobertaTokenizerFast.from_pretrained("./offline_models/roberta-base")
    
    image_processor = AutoImageProcessor.from_pretrained("./offline_models/resnet-152-processor")
    image_processor.size["shortest_edge"] = 512
    image_processor.crop_size = {"height": 512, "width": 512}

    your_dataset_mean = [0.47131237, 0.47472394, 0.43935284]
    your_dataset_std = [0.19183987, 0.18266639, 0.17506209]
    image_processor.image_mean = your_dataset_mean
    image_processor.image_std = your_dataset_std

    if 'height' in image_processor.size: del image_processor.size['height']
    if 'width' in image_processor.size: del image_processor.size['width']
    
    print(f"✅ Image processor set to: {image_processor.size}")

    # === 准备数据路径 ===
    LR_images_path = os.path.join(data_path, "Images_HR")
    
    # 训练集路径 (用于获取 Answer Mapping)
    LR_questionsJSON = os.path.join(data_path, "USGS_split_train_questions.json")
    LR_answersJSON = os.path.join(data_path, "USGS_split_train_answers.json")
    LR_imagesJSON = os.path.join(data_path, "USGS_split_train_images.json")
    
    # 测试集路径
    LR_questionstestJSON = os.path.join(data_path, "USGS_split_test_phili_questions.json")
    LR_answerstestJSON = os.path.join(data_path, "USGS_split_test_phili_answers.json")
    LR_imagestestJSON = os.path.join(data_path, "USGS_split_test_phili_images.json")

    # === 获取 Answer Mapping ===
    print("正在从训练集获取答案映射...")
    temp_train_loader = VQALoader(
        LR_images_path, LR_imagesJSON, LR_questionsJSON, LR_answersJSON,
        tokenizer=tokenizer, image_processor=image_processor, 
        Dataset="HR", train=True
    )
    answer_mapping = temp_train_loader.selected_answers
    print(f"答案映射获取成功，共 {len(answer_mapping)} 个类别。")

    # === 加载测试集 ===
    test_dataset = VQALoader(
        LR_images_path, LR_imagestestJSON, LR_questionstestJSON, LR_answerstestJSON,
        tokenizer=tokenizer, image_processor=image_processor,
        Dataset="HR", train=False, sequence_length=sequence_length,
        selected_answers=answer_mapping # 必须传这个，否则 label id 会乱
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print(f"测试数据加载器准备就绪，共 {len(test_dataset)} 条数据。")

    # === 初始化模型结构 ===
    print("正在初始化模型架构...")
    
    # 1. 实例化模型 (此时是随机初始化的)
    # shared_encoder 只要结构对就行，权重后面会被 ckpt 覆盖
    shared_encoder = RobertaModel.from_pretrained("./offline_models/roberta-base")
    
    model_to_test = VQAModel(
        batch_size=batch_size, 
        lr=1e-5, # 虽然测试用不到 lr，但如果 init 需要就填上
        number_outputs=NUM_CLASS,
        shared_encoder=shared_encoder
    )
    print("模型架构初始化完成。")

    # === 执行评估 ===
    evaluate(
        model=model_to_test,
        test_loader=test_loader,
        device=device,
        checkpoint_path=checkpoint_path
    )