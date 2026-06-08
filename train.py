import psutil
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import torch

random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, [2]))
os.environ["WANDB_INIT_TIMEOUT"] = "600"
os.environ["WANDB_DEBUG"] = "true"
os.environ["WANDB_CORE_DEBUG"] = "true"
os.environ["WANDB_API_KEY"] = (
    "fb8108b4032a87e3e3019d7dd0a9607d5e183009"  # 将引号内的*替换成自己在wandb上的key
)
os.environ["WANDB_MODE"] = "offline"
torch.manual_seed(42)
import typer
import pytorch_lightning as pl
import torchvision.transforms as transforms
#from augment.aug_lr import AutoAugment
#from transformers import RobertaTokenizerFast, ViltImageProcessor
from transformers import RobertaTokenizerFast, AutoImageProcessor
from torchvision.models import resnet152, ResNet152_Weights
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from model.model_main_wt1 import VQAModel, ManualLRStepCallback
from dataloader.VQALoader_lr152_new import VQALoader


torch.set_float32_matmul_precision("highest")


def main(
    num_workers: int = 12,
    ratio_images_to_use: float = 1,
    sequence_length: int = 40,
    num_epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-5,
    Dataset="LR",
):

    data_path = "RSVQA_LR"
    LR_questionsJSON = os.path.join(data_path, "all_questions.json")
    LR_answersJSON = os.path.join(data_path, "all_answers.json")
    LR_imagesJSON = os.path.join(data_path, "LR_split_train_images0.json")
    LR_questionsvalJSON = os.path.join(data_path, "LR_split_val_questions.json")
    LR_answersvalJSON = os.path.join(data_path, "LR_split_val_answers.json")
    LR_imagesvalJSON = os.path.join(data_path, "LR_split_val_images.json")
    LR_images_path = os.path.join(data_path, "Images_LR")

    #tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizerFast.from_pretrained("./offline_models/roberta-base")
    # image_processor = ViltImageProcessor(
    #     do_resize=True,
    #     image_std=[0.229, 0.224, 0.225],
    #     image_mean=[0.485, 0.456, 0.406],
    #     do_rescale=True,
    #     do_normalize=True,
    #     size=512,
    #     size_divisor=32,
    # )
     # ------------------- 添加以下新代码 -------------------
    print("✅ Loading AutoImageProcessor for ResNet-152...")
    # 1. 从预训练模型ID自动加载匹配的处理器
    #    它会自动包含正确的均值、标准差、缩放等所有配置
    #image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-152")
    print("✅ Loading AutoImageProcessor for ResNet-152 from local files...")
    # 1. 从本地文件夹加载处理器配置
    image_processor = AutoImageProcessor.from_pretrained("./offline_models/resnet-152-processor")
    # 2. (关键) 覆盖默认尺寸以匹配您的需求 (512x512)
    #    ResNet152 默认尺寸较小(如224x224)，我们需要手动调整
    # 2. 设置【缩放】的目标：告诉处理器将图像的最短边缩放到 512 像素
    image_processor.size["shortest_edge"] = 256

    # 3. 设置【裁剪】的目标：告诉处理器在缩放后，从中心裁剪出一个 512x512 的图像
    image_processor.crop_size = {"height": 256, "width": 256}

    your_dataset_mean = [0.19952762, 0.26360846, 0.28173736]
    your_dataset_std = [0.08373761, 0.05542858, 0.045889754]


    image_processor.image_mean = your_dataset_mean
    image_processor.image_std = your_dataset_std

    # 4. (可选但推荐) 清理掉之前添加的、现在会引起冲突的键，确保万无一失
    if 'height' in image_processor.size:
        del image_processor.size['height']
    if 'width' in image_processor.size:
        del image_processor.size['width']
    print(f"✅ Image processor size overridden to: {image_processor.size}")
    # ----------------------------------------------------
    # --- 修改结束 ---

    if Dataset == "LR":
        model = VQAModel(
            batch_size=batch_size, lr=lr, number_outputs=9, backbone_name="resnet152"
        )
    else:
        model = VQAModel(
            batch_size=batch_size, lr=lr, number_outputs=94, backbone_name="resnet152"
        )


    # loader for the training data
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
        # transform=transform_train,
    )

    LR_train_loader = torch.utils.data.DataLoader(
        LR_data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # loader for the validation data
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

    LR_val_loader = torch.utils.data.DataLoader(
        LR_data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # 在初始化 WandbLogger 时加入 log_model 和 config 参数
    wandb_logger = WandbLogger(
        project="main_lr",
        name="fusion152_wt_lr_12",
        log_model=True,
        config={
            "batch_size": batch_size,
            "lr": lr,
            "epochs": num_epochs,
            "image_ratio": ratio_images_to_use,
        },
    )

    # specify how to checkpoint
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="valid_AA",
        save_weights_only=True,
        mode="max",
        dirpath="checkpoint_main_wt",
        filename=f"{{epoch}}_{{valid_AA:.5f}}",
    )

    # early stopping
    early_stopping = EarlyStopping(monitor="valid_AA", patience=10, mode="max")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        devices=1,
        accelerator="cuda",
        fast_dev_run=False,
        precision="16-mixed",
        max_epochs=num_epochs,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        # strategy='ddp_find_unused_parameters_true',
        #accumulate_grad_batches=2,  # 新增参数：每2个小batch累积一次梯度
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
    )


    trainer.fit(model, train_dataloaders=LR_train_loader, val_dataloaders=LR_val_loader)


if __name__ == "__main__":
    typer.run(main)
