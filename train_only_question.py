import os
import torch
import numpy as np
import random
from tqdm import tqdm
import time
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl

random.seed(42)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import RobertaModel, RobertaTokenizerFast, AutoImageProcessor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.model_only_question import Model_Q
from dataloader.VQALoader import VQALoader

# 定义与联合训练完全一致的全局变量
num_workers = 12


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_fn(worker_id):
    # 严格保持与联合训练一致的初始化
    np.random.seed(int(seed) + worker_id)
    
    
class ManualLRStepScheduler:
    """
    一个完全手动的、按 Epoch 分阶段设置学习率的原生 PyTorch 调度器。
    适配纯手动 for epoch 训练循环。
    """
    def __init__(self, optimizer, base_lr=1e-5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        
        # ✨ 完整保留您想要的学习率计划 (Epoch 从 0 开始)
        self.lr_schedule = [
            # 阶段 1: Warmup
            (0, 0, self.base_lr * 1.0),      # Epoch 0: 1e-5
            (1, 1, self.base_lr * 2.0),      # Epoch 1: 2e-5
            (2, 3, self.base_lr * 1.0),      # Epoch 2-3: 1e-5
            
            # 阶段 2: Cooldown
            (4, 5, self.base_lr * 0.2),      # Epoch 4-5: 2e-6
            
            # 阶段 3: 第一个衰减点
            (6, 7, self.base_lr * 0.2 * 0.2),# Epoch 6-7: 4e-7
            
            # 阶段 4: 第二个衰减点
            (8, 9, self.base_lr * 0.01),     # Epoch 8-9: 1e-7
            (10, 200, self.base_lr * 0.001)  # Epoch 10+: 1e-8
        ]
        
        # 将计划转换为更高效的查询字典
        self.lr_map = {}
        for (start, end, lr_val) in self.lr_schedule:
            for epoch in range(start, end + 1):
                self.lr_map[epoch] = lr_val

    def step(self, epoch):
        """
        在每轮 Epoch 开始或结束时调用，根据当前 epoch 传入并更新优化器的学习率。
        """
        # 从计划表中查找当前轮次的 LR，如果超出范围则默认保持最后一档
        new_lr = self.lr_map.get(epoch, self.base_lr * 0.001)
        
        # 真正地更新 PyTorch 优化器内部的 param_groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        print(f"\n⚡【手动调度器激活】 Epoch {epoch}: 优化器学习率成功调整为 -> {new_lr:.8f}")


def train_question_only(
    VQA_Q,
    LR_data_train,
    LR_data_val,
    batch_size,
    num_epochs,
    device,
):
    # 梯度累积设置与联合训练一致
    accumulate_grad_batches = 1
    
    LR_train_loader = torch.utils.data.DataLoader(
        LR_data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=_init_fn
    )
    LR_val_loader = torch.utils.data.DataLoader(
        LR_data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=_init_fn
    )
    
    VQA_Q.to(device)

    # --- 优化器设置 ---
    # 仅针对问题模型自有的全部参数进行优化
    optimizer = torch.optim.AdamW(
        VQA_Q.parameters(),
        lr=1e-5,
        weight_decay=1e-4  # 应用相同的权重衰减
    )
    print("优化器已成功配置为 AdamW (仅针对问题模型)。")
    
    # 学习率调度器更换为自定义的智能调度器
    lr_factors = [0.5]
    scheduler = ManualLRStepScheduler(optimizer, base_lr=1e-5)
    print("学习率调度器已配置为 VariableFactorPlateauScheduler。")
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # EarlyStopping 的参数
    early_stopping_patience = 10
    epochs_without_improvement = 0
    
    # ModelCheckpoint 的参数
    save_top_k = 5
    top_k_checkpoints = [] # 用于追踪 top_k 模型 (分数, 路径)
    
    # 用于记录所有轮次历史指标的列表
    all_train_losses, all_valid_losses = [], []
    all_train_accs, all_valid_accs = [], []
    all_learning_rates = []
    
    # 用于追踪最佳模型的变量
    best_valid_AA = 0.0
    best_valid_OA = 0.0
    best_acc_rural_urban, best_acc_presence, best_acc_count, best_acc_comp = 0.0, 0.0, 0.0, 0.0
    
    for epoch in range(num_epochs):
        scheduler.step(epoch)
        VQA_Q.train()
        running_loss = 0
        train_correct, train_total = 0, 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(
            enumerate(LR_train_loader),
            total=len(LR_train_loader),
            desc=f"Epoch {epoch}",
        )
        
        for i, data in progress_bar:
            # 纯文本训练不需要图片，通过解包忽略第一个返回值 pixel_values
            _, input_ids, token_type_ids, attention_mask, answer = data
            
            # 将数据移动到设备
            question_id = input_ids.long().to(device)
            question_seg = token_type_ids.long().to(device)
            question_mask = attention_mask.long().to(device)
            answer = answer.long().to(device)
            
            # 前向传播和损失计算 (仅包含仅问题模型的单任务标准交叉熵)
            pred_q = VQA_Q(question_id, question_seg, question_mask)
            total_loss = criterion(pred_q, answer)
            
            # 梯度累积逻辑
            loss = total_loss / accumulate_grad_batches
            loss.backward()
            
            if (i + 1) % accumulate_grad_batches == 0 or (i + 1) == len(LR_train_loader):
                optimizer.step()      # 更新模型权重
                optimizer.zero_grad() # 清空累积的梯度
            
            running_loss += total_loss.item()
            
            # 计算训练集的准确率
            train_preds = torch.argmax(pred_q, dim=1)
            train_correct += (train_preds == answer).sum().item()
            train_total += answer.size(0)
            progress_bar.set_postfix({"Loss": total_loss.item()})

        print(f"Epoch {epoch} 完成. 平均训练损失: {running_loss / len(LR_train_loader):.4f}")
        avg_train_loss = running_loss / len(LR_train_loader)
        avg_train_acc = train_correct / train_total if train_total > 0 else 0
        
        VQA_Q.eval()
        val_running_loss = 0
        test_step_outputs = []
        
        total_rural_urban, total_presence, total_count, total_comp = 0, 0, 0, 0
        right_rural_urban, right_presence, right_count, right_comp = 0, 0, 0, 0

        with torch.no_grad():
            val_progress_bar = tqdm(LR_val_loader, desc=f"Epoch {epoch} [Validate]")
            for data in val_progress_bar:
                # 同样在验证集循环中解包并解耦图片数据
                (
                    _,
                    input_ids,
                    token_type_ids,
                    attention_mask,
                    answer,
                    question_type,
                    img_id,
                    question,
                    answer_str,
                ) = data
                
                question_id = input_ids.long().to(device)
                question_seg = token_type_ids.long().to(device)
                question_mask = attention_mask.long().to(device)
                answer = answer.long().to(device)
                
                # 模型预测
                pred_q = VQA_Q(question_id, question_seg, question_mask)
                loss = criterion(pred_q, answer)
                val_running_loss += loss.item()
                
                # 统计正确答案
                answer_cpu = answer.cpu().numpy()
                pred_arg = np.argmax(pred_q.cpu().detach().numpy(), axis=1)
                for j in range(pred_arg.shape[0]):
                    if pred_arg[j] == answer_cpu[j]:
                        test_step_outputs.append([1, question_type[j]])
                    else:
                        test_step_outputs.append([0, question_type[j]])
                        
        avg_valid_loss = val_running_loss / len(LR_val_loader) if len(LR_val_loader) > 0 else 0
        
        # 计算 AA, OA, 和各类别准确率
        outputs = np.array(test_step_outputs)
        for j in range(outputs.shape[0]):
            is_correct = int(outputs[j][0])
            q_type = outputs[j][1]
            if q_type == "comp":
                total_comp += 1
                right_comp += is_correct
            elif q_type == "presence":
                total_presence += 1
                right_presence += is_correct
            elif q_type == "count":
                total_count += 1
                right_count += is_correct
            else: # rural_urban
                total_rural_urban += 1
                right_rural_urban += is_correct

        acc_rural_urban = right_rural_urban / total_rural_urban if total_rural_urban > 0 else 0
        acc_presence = right_presence / total_presence if total_presence > 0 else 0
        acc_count = right_count / total_count if total_count > 0 else 0
        acc_comp = right_comp / total_comp if total_comp > 0 else 0
        
        right = right_rural_urban + right_presence + right_count + right_comp
        total = total_rural_urban + total_presence + total_count + total_comp
        
        AA = (acc_rural_urban + acc_presence + acc_count + acc_comp) / 4
        OA = right / total if total > 0 else 0
        

        # 收集并存储本轮的所有指标
        current_lr = optimizer.param_groups[0]['lr']
        all_train_losses.append(avg_train_loss)
        all_valid_losses.append(avg_valid_loss)
        all_train_accs.append(avg_train_acc)
        all_valid_accs.append(OA) # 使用 OA 作为验证准确率记录
        all_learning_rates.append(current_lr)
        
        # 打印所有轮次的历史记录总结
        print("\n--- All Epochs Summary (Question-Only) ---")
        for idx, (train_l, valid_l, train_a, valid_a, lr_rate) in enumerate(zip(all_train_losses, all_valid_losses, all_train_accs, all_valid_accs, all_learning_rates)):
            print(f"Epoch {idx+1}: Train Loss={train_l:.4f}, Valid Loss={valid_l:.4f}, Train Acc={train_a:.4f}, Valid Acc={valid_a:.4f}, LR={lr_rate:.8f}")
        print("------------------------------------------\n")

        # 打印本轮的详细指标
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] Current Epoch {epoch+1} Task Accuracies:")
        print(f"  - OA: {OA:.4f}")
        print(f"  - AA: {AA:.4f}")
        print(f"  - R/U: {acc_rural_urban:.4f}, Presence: {acc_presence:.4f}, Count: {acc_count:.4f}, Comp: {acc_comp:.4f}")
        
        log_dict = {
            "epoch": epoch, "train_loss": avg_train_loss, "train_acc": avg_train_acc,
            "valid_loss": avg_valid_loss, "valid_OA": OA, "valid_AA": AA,
            "learning_rate": current_lr, "val_acc/rural_urban": acc_rural_urban,
            "val_acc/presence": acc_presence, "val_acc/count": acc_count, "val_acc/comp": acc_comp
        }
        wandb.log(log_dict)
        
        save_dir = "checkpoint_q_only_hr"
        os.makedirs(save_dir, exist_ok=True)
        
        # 依据 OA 保存当前最好的纯文本 baseline 模型
        if OA > best_valid_OA:
            print(f"✨ New best OA for Q-Only: {OA:.4f} (previously {best_valid_OA:.4f}). Saving model...")
            best_valid_AA = AA
            best_valid_OA = OA
            best_acc_rural_urban = acc_rural_urban
            best_acc_presence = acc_presence
            best_acc_count = acc_count
            best_acc_comp = acc_comp
            epochs_without_improvement = 0
            best_model_path = "best_model_q_only_hr.ckpt"
            torch.save(VQA_Q.state_dict(), best_model_path)
            print(f"✨ New best Q-only model saved to {best_model_path}!")
        else:
            epochs_without_improvement += 1
            print(f"\nValidation OA did not improve for {epochs_without_improvement} epoch(s). Best is still {best_valid_OA:.4f}.")
        
        current_score_aa = AA
        
        # 检查当前分数是否值得保存到 top-k 列表 (Top-K 维持以 AA 为准)
        should_save = False
        if len(top_k_checkpoints) < save_top_k:
            should_save = True
        elif current_score_aa > min(top_k_checkpoints, key=lambda x: x[0])[0]:
            should_save = True

        if should_save:
            if len(top_k_checkpoints) >= save_top_k:
                worst_checkpoint = min(top_k_checkpoints, key=lambda x: x[0])
                if os.path.exists(worst_checkpoint[1]):
                    os.remove(worst_checkpoint[1])
                top_k_checkpoints.remove(worst_checkpoint)

            filename = f"epoch={epoch+1}_valid_AA={current_score_aa:.5f}.ckpt"
            save_path = os.path.join(save_dir, filename)
            torch.save(VQA_Q.state_dict(), save_path)
            top_k_checkpoints.append((current_score_aa, save_path))
            print(f"✨ Saved new top-{save_top_k} model (by AA): {filename}")

        # 早停检查 (基于 OA)
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n❗️ Early stopping triggered after {early_stopping_patience} epochs without OA improvement.")
            break
            
        print(f"\n[Summary] Best so far -> AA: {best_valid_AA:.4f}, OA: {best_valid_OA:.4f}")
        print(f"     ↳ R/U: {best_acc_rural_urban:.4f}, Presence: {best_acc_presence:.4f}, Count: {best_acc_count:.4f}, Comp: {best_acc_comp:.4f}\n")
        
    # 所有训练轮次结束后，执行绘图逻辑
    print("\n--- Training finished. Plotting and saving loss curve... ---")
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label="Training Loss")
    plt.plot(all_valid_losses, label="Validation Loss")
    plt.title("Q-Only Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plot_dir = "loss_plots_q_only"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plot_path = os.path.join(plot_dir, "loss_curve_q_only_hr.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Loss curve saved to: {plot_path}")


if __name__ == "__main__":

    seed = 42
    seed_torch(seed)
    Dataset = "LR"
    modeltype = "Simple"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    batch_size = 32
    num_epochs = 200
    patch_size = 512
    ratio_images_to_use = 1
    lr = 1e-5
    sequence_length = 40

    data_path = "RSVQA_HR"
    LR_questionsJSON = os.path.join(data_path, "USGS_split_train_questions.json")
    LR_answersJSON = os.path.join(data_path, "USGS_split_train_answers.json")
    LR_imagesJSON = os.path.join(data_path, "USGS_split_train_images.json")
    LR_questionsvalJSON = os.path.join(data_path, "USGS_split_val_questions.json")
    LR_answersvalJSON = os.path.join(data_path, "USGS_split_val_answers.json")
    LR_imagesvalJSON = os.path.join(data_path, "USGS_split_val_images.json")
    LR_images_path = os.path.join(data_path, "Images_HR")

    # 在这里初始化 wandb 实验
    os.environ["WANDB_API_KEY"] = "fb8108b4032a87e3e3019d7dd0a9607d5e183009"
    os.environ["WANDB_MODE"] = "offline" 

    wandb.init(
        project="main_lr",
        name="question_only_baseline_2026", 
        config={
            "batch_size": batch_size,
            "lr": lr,
            "epochs": num_epochs,
            "architecture": "Question-Only Manual Loop", 
            "dataset": Dataset,
        }
    )

    print("Step1 is ok")

    tokenizer = RobertaTokenizerFast.from_pretrained("./offline_models/roberta-base")
    print("✅ Loading AutoImageProcessor for ResNet-152 from local files...")
    image_processor = AutoImageProcessor.from_pretrained("./offline_models/resnet-152-processor")
    
    image_processor.size["shortest_edge"] = 512
    image_processor.crop_size = {"height": 512, "width": 512}

    your_dataset_mean = [0.47131237, 0.47472394, 0.43935284]
    your_dataset_std = [0.19183987, 0.18266639, 0.17506209]

    image_processor.image_mean = your_dataset_mean
    image_processor.image_std = your_dataset_std

    if 'height' in image_processor.size:
        del image_processor.size['height']
    if 'width' in image_processor.size:
        del image_processor.size['width']
    print(f"✅ Image processor size overridden to: {image_processor.size}")
    print("Step2 is ok")

    # 构建数据加载器 (完全保留相同的 image_processor 传入，确保 DataLoader 行为一致性)
    LR_data_train = VQALoader(
        LR_images_path,
        LR_imagesJSON,
        LR_questionsJSON,
        LR_answersJSON,
        tokenizer=tokenizer,
        image_processor=image_processor,
        Dataset="LR",
        train=True,
        sequence_length=sequence_length,
        ratio_images_to_use=ratio_images_to_use,
    )
    LR_data_val = VQALoader(
        LR_images_path,
        LR_imagesvalJSON,
        LR_questionsvalJSON,
        LR_answersvalJSON,
        tokenizer=tokenizer,
        image_processor=image_processor,
        Dataset="LR",
        train=False,
        ratio_images_to_use=1,
        sequence_length=sequence_length,
        selected_answers=LR_data_train.selected_answers,
    )

    print("Step3 is ok")
    print("第四步: 正在加载独立的 RoBERTa 模型...")
    shared_encoder = RobertaModel.from_pretrained("./offline_models/roberta-base")
    print("编码器加载成功。")

    print("正在初始化 Model_Q...")
    model_vqa_q = Model_Q(
        batch_size=batch_size,
        lr=lr,
        number_outputs=9,
        shared_encoder=shared_encoder,
    )

    print("Step4 is ok")

    train_question_only(
        model_vqa_q,
        LR_data_train,
        LR_data_val,
        batch_size,
        num_epochs,
        device,
    )
    print("Over")