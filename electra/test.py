import torch
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from dataset import get_dataloader
from model import primary_encoder, weighting_network
import config as train_config
from easydict import EasyDict as edict
import loss
import torch.nn as nn
from util import save_checkpoint,one_hot,iter_product,clip_gradient,load_model

def test(epoch, test_loader, model_main, model_helper, loss_function, log):
    model_main.eval()
    model_helper.eval()
    
    test_loss = 0
    total_emo_pred_1, total_emo_pred_2, total_emo_true = [], [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            text_name = "review" if "sst-" in log.param.dataset else "cause"
            label_name = "sentiment" if "sst-" in log.param.dataset else "emotion"

            text = batch[text_name]
            attn = batch[f"{text_name}_attn_mask"]
            emotion = torch.tensor(batch[label_name]).long()

            if torch.cuda.is_available():
                text, attn, emotion = text.cuda(), attn.cuda(), emotion.cuda()

            emo_pred_1, _ = model_main(text, attn)
            emo_pred_2 = model_helper(text, attn)

            emo_pred_list_1 = torch.max(emo_pred_1, 1)[1].cpu().tolist()
            emo_pred_list_2 = torch.max(emo_pred_2, 1)[1].cpu().tolist()
            emo_true_list = emotion.cpu().tolist()

            total_emo_pred_1.extend(emo_pred_list_1)
            total_emo_pred_2.extend(emo_pred_list_2)
            total_emo_true.extend(emo_true_list)

    f1_score_emo_1 = f1_score(total_emo_true, total_emo_pred_1, average="macro")
    f1_score_emo_2 = f1_score(total_emo_true, total_emo_pred_2, average="macro")
    
    acc_1 = accuracy_score(total_emo_true, total_emo_pred_1)
    acc_2 = accuracy_score(total_emo_true, total_emo_pred_2)

    print("Classification Report (Model 1):\n", classification_report(total_emo_true, total_emo_pred_1))
    print("Classification Report (Model 2):\n", classification_report(total_emo_true, total_emo_pred_2))

    print("Confusion Matrix (Model 1):\n", confusion_matrix(total_emo_true, total_emo_pred_1))
    print("Confusion Matrix (Model 2):\n", confusion_matrix(total_emo_true, total_emo_pred_2))

    # Display sample inputs and predictions
    print("Sample Inputs and Predictions (Model 1):")
    for i in range(min(5, len(total_emo_true))):
        print(f"True: {total_emo_true[i]}, Predicted: {total_emo_pred_1[i]}")

    print("Sample Inputs and Predictions (Model 2):")
    for i in range(min(5, len(total_emo_true))):
        print(f"True: {total_emo_true[i]}, Predicted: {total_emo_pred_2[i]}")

    return acc_1, acc_2, f1_score_emo_1, f1_score_emo_2

def evaluate_models(log):
    _, _, test_data = get_dataloader(
        log.param.batch_size,
        log.param.dataset,
        w_aug=False,
        label_list=log.param.label_list
    )

    model_main = primary_encoder(
        log.param.batch_size,
        log.param.hidden_size,
        log.param.emotion_size,
        log.param.model_type
    )
    model_helper = weighting_network(
        log.param.batch_size,
        log.param.hidden_size,
        log.param.emotion_size,
        log.param.model_type
    )

    checkpoint_dir = "C:/Users/ADMIN/Downloads/save_model"
    model_main.load_state_dict(torch.load(f"{checkpoint_dir}/model_main_best.pth")['state_dict'])
    model_helper.load_state_dict(torch.load(f"{checkpoint_dir}/model_helper_best.pth")['state_dict'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_main.to(device)
    model_helper.to(device)

    losses = nn.CrossEntropyLoss()

    epoch = 1
    acc_1, acc_2, f1_1, f1_2 = test(
        epoch, test_data, model_main, model_helper, losses, log
    )

    print("Accuracy (Model 1):", acc_1)
    print("Accuracy (Model 2):", acc_2)
    print("F1 Score (Model 1):", f1_1)
    print("F1 Score (Model 2):", f1_2)

if __name__ == "__main__":
    tuning_param = train_config.tuning_param
    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list))

    for param_com in param_list[1:]:
        log = edict()
        log.param = train_config.param

        for num, val in enumerate(param_com):
            log.param[param_list[0][num]] = val
        if log.param.dataset == "sst-5":
            log.param.emotion_size = 5

        evaluate_models(log)
