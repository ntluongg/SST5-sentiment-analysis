import torch
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from util import save_checkpoint, load_model
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
    total_epoch_acc_1, total_epoch_acc_2 = 0, 0
    total_emo_pred_1, total_emo_pred_2, total_emo_true = [], [], []
    total_pred_prob_1, total_pred_prob_2 = [], []
    acc_curve_1, acc_curve_2 = [], []
    total_feature = []
    
    save_pred = {
        "true": [],
        "pred_1": [],
        "pred_2": [],
        "pred_prob_1": [],
        "pred_prob_2": [],
        "feature": []
    }

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            text_name = "review" if "sst-" in log.param.dataset else "cause"
            label_name = "sentiment" if "sst-" in log.param.dataset else "emotion"

            text = batch[text_name]
            attn = batch[f"{text_name}_attn_mask"]
            emotion = torch.tensor(batch[label_name]).long()

            if torch.cuda.is_available():
                text, attn, emotion = text.cuda(), attn.cuda(), emotion.cuda()

            emo_pred_1, supcon_feature_1 = model_main(text, attn)
            emo_pred_2 = model_helper(text, attn)

            num_corrects_1 = (torch.max(emo_pred_1, 1)[1].view(emotion.size()) == emotion).float().sum()
            acc_1 = 100.0 * num_corrects_1 / len(emotion)
            acc_curve_1.append(acc_1.item())
            total_epoch_acc_1 += acc_1.item()

            num_corrects_2 = (torch.max(emo_pred_2, 1)[1].view(emotion.size()) == emotion).float().sum()
            acc_2 = 100.0 * num_corrects_2 / len(emotion)
            acc_curve_2.append(acc_2.item())
            total_epoch_acc_2 += acc_2.item()

            emo_pred_list_1 = torch.max(emo_pred_1, 1)[1].view(emotion.size()).cpu().tolist()
            emo_pred_list_2 = torch.max(emo_pred_2, 1)[1].view(emotion.size()).cpu().tolist()
            emo_true_list = emotion.cpu().tolist()

            total_emo_pred_1.extend(emo_pred_list_1)
            total_emo_pred_2.extend(emo_pred_list_2)
            total_emo_true.extend(emo_true_list)
            total_feature.extend(supcon_feature_1.cpu().tolist())
            total_pred_prob_1.extend(emo_pred_1.cpu().tolist())
            total_pred_prob_2.extend(emo_pred_2.cpu().tolist())

    f1_score_emo_1 = f1_score(total_emo_true, total_emo_pred_1, average="macro")
    f1_score_emo_2 = f1_score(total_emo_true, total_emo_pred_2, average="macro")

    f1_score_1 = {"macro": f1_score_emo_1}
    f1_score_2 = {"macro": f1_score_emo_2}

    save_pred.update({
        "true": total_emo_true,
        "pred_1": total_emo_pred_1,
        "pred_2": total_emo_pred_2,
        "feature": total_feature,
        "pred_prob_1": total_pred_prob_1,
        "pred_prob_2": total_pred_prob_2
    })

    return total_epoch_acc_1 / len(test_loader), total_epoch_acc_2 / len(test_loader), f1_score_1, f1_score_2, save_pred, acc_curve_1, acc_curve_2

def evaluate_models(log):
    """
    Evaluate the trained models on the test dataset.

    Args:
        log (dict): Configuration dictionary containing model and evaluation parameters.
    """
    _, _, test_data = get_dataloader(
        log.param.batch_size,
        log.param.dataset,
        w_aug=False,
        label_list=log.param.label_list
    )

    model_main = primary_encoder(
        log.param.batch_size,
        log.param.hidden_size,
        5,
        log.param.model_type
    )
    model_helper = weighting_network(
        log.param.batch_size,
        log.param.hidden_size,
        5,
        log.param.model_type
    )

    #checkpoint_dir = "C:/Users/ADMIN/Documents/GitHub/LCL_loss/save/final/sst-5/lcl/57.9"
    checkpoint_dir = "C:/Users/ADMIN/Downloads/save_model"
    model_main.load_state_dict(torch.load(f"{checkpoint_dir}/model_main_best.pth")['state_dict'])
    model_helper.load_state_dict(torch.load(f"{checkpoint_dir}/model_helper_best.pth")['state_dict'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_main.to(device)
    model_helper.to(device)

    losses = {
        "contrastive": loss.LCL(temperature=log.param.temperature),
        "emotion": nn.CrossEntropyLoss(),
        "lambda_loss": log.param.lambda_loss
    }

    epoch = 1
    acc_1, acc_2, f1_1, f1_2, save_pred, acc_curve_1, acc_curve_2 = test(
        epoch, test_data, model_main, model_helper, losses["emotion"], log
    )

    print("Accuracy (Model 1):", acc_1)
    print("Accuracy (Model 2):", acc_2)
    print("F1 Score (Model 1):", f1_1)
    print("F1 Score (Model 2):", f1_2)
    # print("Prediction Results:", json.dumps(save_pred, indent=4))

if __name__ == "__main__":

    tuning_param = train_config.tuning_param
    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val
        if log.param.run_name == "subset":
            log.param.emotion_size = int(log.param.label_list.split("-")[0])
        ## reseeding before every run while tuning

        if log.param.dataset == "ed":
            log.param.emotion_size = 32
        elif log.param.dataset == "emoint":
            log.param.emotion_size = 4
        elif log.param.dataset == "goemotions":
            log.param.emotion_size = 27
        elif log.param.dataset == "isear":
            log.param.emotion_size = 7
        elif log.param.dataset == "sst-2":
            log.param.emotion_size = 2
        elif log.param.dataset == "sst-5":
            log.param.emotion_size = 5

    evaluate_models(log)
