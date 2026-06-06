import os.path
import json
import torch
from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomHorizontalFlip
from .auto_augment import AutoAugment


class VQALoader(Dataset):
    """
    This class manages the Dataloading.
    """

    def __init__(
        self,
        imgFolder,
        images_file,
        questions_file,
        answers_file,
        tokenizer,
        image_processor,
        Dataset,
        train=True,
        ratio_images_to_use=None,
        selected_answers=None,
        sequence_length=40,
        transform=None,
        label=None,
    ):

        self.train = train
        self.imgFolder = imgFolder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        LR_number_outputs = 9
        HR_number_outputs = 94
        self.Dataset = Dataset
        self.transform = transform
        self.dict = {}
        self.spatial_keywords = ["left", "right", "top", "bottom"]
        
        # sequence length of the tokens
        self.sequence_length = sequence_length

        # loading the json files for the question, answers and images
        print("Loading JSONs...")
        with open(questions_file) as json_data:
            questionsJSON = json.load(json_data)

        with open(answers_file) as json_data:
            answersJSON = json.load(json_data)

        with open(images_file) as json_data:
            imagesJSON = json.load(json_data)
        print("Done.")

        # select only the active images
        images = [img["id"] for img in imagesJSON["images"] if img["active"]]

        # select the requested amount of images
        images = images[:int(len(images))]
        self.img_ids = images
        
        if self.Dataset == "LR":
            self.image_paths = [
                os.path.join(imgFolder, str(image) + ".tif") for image in images
            ]
        else:
            self.image_paths = [
                os.path.join(imgFolder, str(image) + ".tif") for image in images
            ]

        print("Construction of the Dataset")

        # when training we construct the answer set
        if train:
            self.freq_dict = {}

            for i, image in enumerate(tqdm(images)):
                for questionid in imagesJSON["images"][image]["questions_ids"]:
                    question = questionsJSON["questions"][questionid]
                    answer_str = answersJSON["answers"][question["answers_ids"][0]]["answer"]

                    # group the counting answers
                    if self.Dataset == "LR":
                        if answer_str.isdigit():
                            num = int(answer_str)
                            if num > 0 and num <= 10:
                                answer_str = "between 0 and 10"
                            if num > 10 and num <= 100:
                                answer_str = "between 10 and 100"
                            if num > 100 and num <= 1000:
                                answer_str = "between 100 and 1000"
                            if num > 1000:
                                answer_str = "more than 1000"
                    else:
                        if "m2" in answer_str:
                            num = int(answer_str[:-2])
                            if num > 0 and num <= 10:
                                answer_str = "between 0m2 and 10m2"
                            if num > 10 and num <= 100:
                                answer_str = "between 10m2 and 100m2"
                            if num > 100 and num <= 1000:
                                answer_str = "between 100m2 and 1000m2"
                            if num > 1000:
                                answer_str = "more than 1000m2"

                    # update the dictionary
                    if answer_str not in self.freq_dict:
                        self.freq_dict[answer_str] = 1
                    else:
                        self.freq_dict[answer_str] += 1

            # sort the dictionary by the most common
            self.freq_dict = sorted(
                self.freq_dict.items(), key=lambda x: x[1], reverse=True
            )

            self.selected_answers = []
            self.non_selected_answers = []

            coverage = 0
            total_answers = 0

            for i, key in enumerate(self.freq_dict):
                if self.Dataset == "LR":
                    if i < LR_number_outputs:
                        self.selected_answers.append(key[0])
                        coverage += key[1]
                    else:
                        self.non_selected_answers.append(key[0])
                    total_answers += key[1]
                else:
                    if i < HR_number_outputs:
                        self.selected_answers.append(key[0])
                        coverage += key[1]
                    else:
                        self.non_selected_answers.append(key[0])
                    total_answers += key[1]

        else:
            self.selected_answers = selected_answers


        self.images_questions_answers = []


        for i, image in enumerate(tqdm(images)):
            for questionid in imagesJSON["images"][image]["questions_ids"]:
                question = questionsJSON["questions"][questionid]
                question_str = question["question"]
                type_str = question["type"]
                answer_str = answersJSON["answers"][question["answers_ids"][0]]["answer"]

                if self.Dataset == "LR":
                    if answer_str.isdigit():
                        num = int(answer_str)
                        if num > 0 and num <= 10:
                            answer_str = "between 0 and 10"
                        if num > 10 and num <= 100:
                            answer_str = "between 10 and 100"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100 and 1000"
                        if num > 1000:
                            answer_str = "more than 1000"
                else:
                    if "m2" in answer_str:
                        num = int(answer_str[:-2])
                        if num > 0 and num <= 10:
                            answer_str = "between 0m2 and 10m2"
                        if num > 10 and num <= 100:
                            answer_str = "between 10m2 and 100m2"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100m2 and 1000m2"
                        if num > 1000:
                            answer_str = "more than 1000m2"

                if answer_str in self.selected_answers:
                    answer = self.selected_answers.index(answer_str)
                    if label is not None:
                        if type_str == label:
                            self.images_questions_answers.append(
                                [question_str, answer, i, type_str, answer_str]
                            )
                    else:
                        self.images_questions_answers.append(
                            [question_str, answer, i, type_str, answer_str]
                        )

        print("Done.")

        if self.train:
            self.auto_augment = AutoAugment()
            self.train_transform = self.auto_augment  # 原有的 AutoAugment
            print("✅ VQALoader: 启用 AutoAugment 数据增强 (SOTA 策略)")
        
    def __len__(self):
        return len(self.images_questions_answers)

    def __getitem__(self, idx):
        # load the features of the index
        data = self.images_questions_answers[idx]
        question_text = data[0]

        is_spatial_question = any(
            keyword in question_text for keyword in self.spatial_keywords
        )

        language_feats = self.tokenizer(
            question_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.sequence_length,
        )
        
        img_path = self.image_paths[data[2]]
        img = io.imread(img_path) 
        
        img_pil = Image.fromarray(img)

        if self.train:
            if random.random() < 0.55:
                img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.55:
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:  
                rot_method = random.choice([
                    Image.ROTATE_90, 
                    Image.ROTATE_180, 
                    Image.ROTATE_270
                ])
                img_pil = img_pil.transpose(rot_method)
            img_pil = self.train_transform(img_pil)

        imgT = self.image_processor(img_pil, return_tensors="pt")
        

        if self.train:
            token_type_ids = torch.zeros_like(language_feats["input_ids"][0])
            return (
                imgT["pixel_values"][0],
                language_feats["input_ids"][0],
                token_type_ids,
                language_feats["attention_mask"][0],
                data[1],
            )
        else:
            token_type_ids = torch.zeros_like(language_feats["input_ids"][0])
            return (
                imgT["pixel_values"][0],
                language_feats["input_ids"][0],
                token_type_ids,
                language_feats["attention_mask"][0],
                data[1],
                data[3],
                data[2],
                data[0],
                data[-1],
            )
