import os
import random
import sys
from logging import getLogger

import hydra
import numpy as np
import torch
from seqeval.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForTokenClassification

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from bpe_dropout import RobertaTokenizerDropout
from utils import boi1_to_2, get_label, path_to_data, save_choices, val_to_key

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


def get_inputs(
    text,
    labels,
    indexes,
    idx_start,
    idx_end,
    tokenizer,
    padding,
    post_sentence_padding,
    add_sep_between_sentences,
):
    max_length = padding - 2 if padding else len(text)

    subwords, masked_ids = tokenizer.tokenzizeSentence(" ".join(text[idx_start:idx_end]))
    row_tokens = text[idx_start:idx_end]
    row_labels = labels[idx_start:idx_end]

    if post_sentence_padding:
        while len(subwords) < max_length and idx_end < len(text):
            if add_sep_between_sentences and idx_end in [d[0] for d in indexes]:
                subwords.append(tokenizer.sep_token)
                masked_ids.append(-100)
            ex_subwords = tokenizer.tokenize(" " + text[idx_end])
            subwords = subwords + ex_subwords
            masked_ids = masked_ids + [-100] * len(ex_subwords)
            idx_end += 1
            if len(subwords) < max_length:
                subwords = subwords[:max_length]
                masked_ids = masked_ids[:max_length]
        subwords = (
            [tokenizer.cls_token_id] + [tokenizer._convert_token_to_id(w) for w in subwords] + [tokenizer.sep_token_id]
        )
        masked_ids = [-100] + masked_ids + [-100]
        if len(subwords) >= padding:
            subwords = subwords[:padding]
            masked_ids = masked_ids[:padding]
            mask = [1] * padding

        else:
            attention_len = len(subwords)
            pad_len = padding - len(subwords)
            subwords += [tokenizer.pad_token_id] * pad_len
            masked_ids += [-100] * pad_len
            mask = [1] * attention_len + [0] * pad_len

        masked_label = row_labels
        masked_label_ids = get_label(masked_ids, masked_label, "PAD")

    data = {
        "input_ids": torch.tensor([subwords], dtype=torch.int),
        "attention_mask": torch.tensor([mask], dtype=torch.int),
        "predict_labels": torch.tensor([masked_label_ids], dtype=torch.long),
        "tokens": row_tokens,
        "labels": row_labels,
    }
    data["token_type_ids"] = torch.zeros_like(data["attention_mask"])
    return data


def pred(
    tokenizer,
    test_dataset,
    model,
    device="cuda",
    length=512,
    loop=500,
    post_sentence_padding=False,
    add_sep_between_sentences=False,
):
    output = []
    with torch.no_grad():
        for test_data in tqdm(test_dataset, leave=False):
            for idx_start, idx_end in tqdm(test_data["doc_index"], leave=False):
                pred_labels = [
                    {
                        "labels": [
                            val_to_key(true_label, ner_dict) for true_label in test_data["labels"][idx_start:idx_end]
                        ],
                        "f1_score": 1.0,
                        "count": 0,
                    }
                ]
                for j in tqdm(range(loop), leave=False):
                    test_inputs = get_inputs(
                        test_data["tokens"],
                        test_data["labels"],
                        test_data["doc_index"],
                        idx_start,
                        idx_end,
                        tokenizer,
                        padding=length,
                        post_sentence_padding=post_sentence_padding,
                        add_sep_between_sentences=add_sep_between_sentences,
                    )
                    input, mask = (
                        test_inputs["input_ids"].to(device),
                        test_inputs["attention_mask"].to(device),
                    )
                    pred = model(input, mask).logits.argmax(-1).to("cpu").tolist()[0]
                    label = test_inputs["predict_labels"][0]

                    pred = [val_to_key(prd, ner_dict) for (prd, lbl) in zip(pred, label) if lbl != ner_dict["PAD"]]
                    pred = [c if c != "PAD" else "O" for c in pred]
                    pred = boi1_to_2(pred)

                    out_token = test_inputs["tokens"]
                    out_label = test_inputs["labels"]
                    out_label = [val_to_key(o_n, ner_dict) for o_n in out_label]

                    label_list = [pl["labels"] for pl in pred_labels]
                    if pred not in label_list:
                        f1 = f1_score([out_label], [pred], zero_division=1)
                        pred_labels.append({"labels": pred, "f1_score": f1, "count": 1})
                    else:
                        idx = label_list.index(pred)
                        pred_labels[idx]["count"] += 1

                out = {
                    "tokens": out_token,
                    "true_label": out_label,
                    "pred_labels": pred_labels,
                }
                output.append(out)

    return output


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

    tokenizer = RobertaTokenizerDropout.from_pretrained(cfg.model_name, alpha=cfg.pred_p)

    local_model = os.path.join(root_path, "model/epoch19.pth")
    model = AutoModelForTokenClassification.from_pretrained(cfg.model_name, num_labels=len(ner_dict))
    model.load_state_dict(torch.load(local_model))
    model = model.to(cfg.device)

    if cfg.test == "test":
        valid_dataset = path_to_data(os.path.join(root_path, "row_data/eng.testb"))

    elif cfg.test == "valid":
        valid_dataset = path_to_data(os.path.join(root_path, "row_data/eng.testa"))

    elif cfg.test == "2023":
        valid_dataset = path_to_data(os.path.join(root_path, "row_data/conllpp.txt"))

    elif cfg.test == "crossweigh":
        valid_dataset = path_to_data(os.path.join(root_path, "row_data/conllcw.txt"))

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
    save_choices(out, valid_dataset, f"choice_data/{cfg.test}_choices.json")


if __name__ == "__main__":
    main()
