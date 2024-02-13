import os
import random
import shutil
import sys
from logging import getLogger

import hydra
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForTokenClassification

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from bpe_dropout import RobertaTokenizerDropout
from predictor import pred
from trainer import train
from utils import path_to_data, save_choices

root_path = os.getcwd()
logger = getLogger(__name__)
ner_dict = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
    "PAD": 9,
}


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visible_devices
    if cfg.huggingface_cache:
        os.environ["TRANSFORMERS_CACHE"] = cfg.huggingface_cache

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True

    train_dataset = path_to_data(os.path.join(root_path, "row_data/eng.train"))
    valid_dataset = path_to_data(os.path.join(root_path, "row_data/eng.testa"))

    train(
        train_datasets=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        num_epoch=cfg.num_epoch,
        length=cfg.length,
        p=cfg.p,
        seed=cfg.seed,
        model_name=cfg.model_name,
        accum_iter=cfg.accum_iter,
        weight_decay=cfg.weight_decay,
        use_loss_weight=cfg.use_loss_weight,
        use_scheduler=cfg.use_scheduler,
        warmup_late=cfg.warmup_late,
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
    )

    tokenizer = RobertaTokenizerDropout.from_pretrained(cfg.model_name, alpha=cfg.pred_p)

    old_model_name = f"./model/epoch{cfg.num_epoch-1}.pth"
    local_model = os.path.join(root_path, "model", f"epoch{cfg.num_epoch-1}.pth")

    shutil.copy(old_model_name, local_model)

    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=len(ner_dict))
    model = AutoModelForTokenClassification.from_config(config)
    model.load_state_dict(torch.load(local_model))
    model = model.to(cfg.device)

    out = pred(
        tokenizer=tokenizer,
        test_dataset=valid_dataset,
        model=model,
        loop=cfg.loop,
        length=cfg.length,
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
        device=cfg.device,
    )
    save_choices(out, valid_dataset, "choice_data/valid_choices.json")


if __name__ == "__main__":
    main()
