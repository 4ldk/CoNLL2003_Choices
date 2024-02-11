import json
import os
import random
import sys
from logging import getLogger

import numpy as np
import seqeval.metrics

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from utils import path_to_data, val_to_key

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


def majority(preds):
    majority = preds[0]
    return majority


def n_th_majority(preds, n=2, choice="f1", f1s=None, counts=None):
    majorities = preds[:n]
    if f1s is not None:
        f1s = f1s[:n]
    if counts is not None:
        counts = counts[:n]
        if counts[-1] == 0:
            majorities = majorities[:-1]
            f1s = f1s[:-1]
            counts = counts[:-1]

    if choice == "f1":
        max_f1_idx = np.argmax(np.array(f1s))
        choiced = majorities[max_f1_idx]
    elif choice == "random":
        choiced = random.choice(majorities)
    elif choice == "weighted_random":
        choiced = random.choices(majorities, weights=counts)[0]
    elif choice == "none":
        choiced = majorities
    return choiced


def main():

    with open(os.path.join(root_path, "choice_data/2023_choices.json")) as f:
        choice_dataset = json.load(f)

    path = os.path.join(root_path, "row_data/conllpp.txt")
    token_dataset = path_to_data(path)

    golden_labels = []
    choiced_labels = []
    for choices in choice_dataset:
        doc_id = choices["document_id"]
        sentence_id = choices["sentence_id"]
        tokens = token_dataset[doc_id]
        sentence_i, sentence_j = tokens["doc_index"][sentence_id]

        golden = [val_to_key(token, ner_dict) for token in tokens["labels"][sentence_i:sentence_j]]
        preds = [c["labels"] for c in choices["pred_labels"]]
        f1s = [c["f1_score"] for c in choices["pred_labels"]]
        counts = [c["count"] for c in choices["pred_labels"]]
        choiced = n_th_majority(preds, n=2, choice="f1", f1s=f1s, counts=counts)

        golden_labels.append(golden)
        choiced_labels.append(choiced)

    print(seqeval.metrics.classification_report(golden_labels, choiced_labels, digits=4))


if __name__ == "__main__":
    main()
