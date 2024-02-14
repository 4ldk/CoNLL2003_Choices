import os
import random
import sys
from logging import getLogger

import hydra
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification
import seqeval.metrics

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from bpe_dropout import RobertaTokenizerDropout
from utils import dataset_encode, get_dataloader, path_to_data, val_to_key

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

    tokenizer = RobertaTokenizerDropout.from_pretrained(cfg.model_name, alpha=cfg.pred_p)

    local_model = os.path.join(root_path, "model/epoch19.pth")
    model = AutoModelForTokenClassification.from_pretrained(cfg.model_name, num_labels=len(ner_dict))
    model.load_state_dict(torch.load(local_model))
    model = model.to(cfg.device)
    model.eval()

    if cfg.test == "test":
        valid_dataset = path_to_data(os.path.join(root_path, "row_data/eng.testb"))

    elif cfg.test == "valid":
        valid_dataset = path_to_data(os.path.join(root_path, "row_data/eng.testa"))

    elif cfg.test == "2023":
        valid_dataset = path_to_data(os.path.join(root_path, "row_data/conllpp.txt"))

    elif cfg.test == "crossweigh":
        valid_dataset = path_to_data(os.path.join(root_path, "row_data/conllcw.txt"))

    encoded_dataset = dataset_encode(
        tokenizer,
        valid_dataset,
        p=0,
        padding=cfg.length,
        subword_label="PAD",
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
    )
    dataloader = get_dataloader(encoded_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    labels, out_labels = (
        encoded_dataset["predict_labels"].tolist(),
        encoded_dataset["labels"],
    )

    golden_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, (input, mask, _, label, _) in enumerate(tqdm(dataloader, leave=False)):
            input, mask = (
                input.to(cfg.device),
                mask.to(cfg.device),
            )
            preds = model(input, mask).logits.argmax(-1).to("cpu").tolist()
            for j, pred in enumerate(preds):
                label = labels[i * cfg.batch_size + j]
                out_label = out_labels[i * cfg.batch_size + j]

                pred = [val_to_key(prd, ner_dict) for (prd, lbl) in zip(pred, label) if lbl != ner_dict["PAD"]]
                pred = [c if c != "PAD" else "O" for c in pred]

                out_label = [val_to_key(o_n, ner_dict) for o_n in out_label]
                golden_labels.append(out_label)
                pred_labels.append(pred)
    print(len(golden_labels), golden_labels[0], "\n", pred_labels[0])
    print(seqeval.metrics.classification_report(pred_labels, golden_labels, digits=4))


if __name__ == "__main__":
    main()
