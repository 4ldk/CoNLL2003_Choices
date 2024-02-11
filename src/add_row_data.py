import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from utils import path_to_data, val_to_key

root_path = os.getcwd()
os.makedirs(os.path.join(root_path, "full_choice_data"), exist_ok=True)

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

for test in ["train", "valid", "test", "2023", "crossweigh"]:

    if test == "train":
        row_dataset = path_to_data(os.path.join(root_path, "row_data/eng.train"))

    elif test == "valid":
        row_dataset = path_to_data(os.path.join(root_path, "row_data/eng.testa"))

    elif test == "test":
        row_dataset = path_to_data(os.path.join(root_path, "row_data/eng.testb"))

    elif test == "2023":
        row_dataset = path_to_data(os.path.join(root_path, "row_data/conllpp.txt"))

    elif test == "crossweigh":
        row_dataset = path_to_data(os.path.join(root_path, "row_data/conllcw.txt"))

    with open(os.path.join(root_path, f"choice_data/{test}_choices.json"), encoding="utf-8") as f:
        choice_dataset = json.load(f)

    output = []
    for choices in choice_dataset:
        doc_id = choices["document_id"]
        sentence_id = choices["sentence_id"]
        row_data = row_dataset[doc_id]
        sentence_i, sentence_j = row_data["doc_index"][sentence_id]

        tokens = row_data["tokens"][sentence_i:sentence_j]
        true_label = [val_to_key(token, ner_dict) for token in row_data["labels"][sentence_i:sentence_j]]

        output.append(
            dict(
                document_id=doc_id,
                sentence_id=sentence_id,
                tokens=tokens,
                true_label=true_label,
                pred_labels=choices["pred_labels"],
            )
        )

    with open(os.path.join(root_path, "full_choice_data", f"{test}_choices.json"), "wt", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
