import os
import pickle
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from utils import path_to_data

root_path = os.getcwd()


cross_var_num = 5

root_path = os.getcwd()
random.seed(42)

save_dir = os.path.join(root_path, "row_data/cross_var")
os.makedirs(save_dir)

train_dataset = path_to_data(os.path.join(root_path, "row_data/eng.train"))

random.shuffle(train_dataset)

var_sentence_num = sum([len(t["doc_index"]) for t in train_dataset]) // cross_var_num
cur_sentence_num = 0
selected_sentence_nums = []
selected_data = []

for train_data in train_dataset:
    cur_sentence_num += len(train_data["doc_index"])
    selected_data.append(train_data)
    if cur_sentence_num > var_sentence_num:
        selected_sentence_nums.append(cur_sentence_num)
        cur_sentence_num = 0
        with open(
            os.path.join(save_dir, f"cross{len(selected_sentence_nums)}.pickle"),
            mode="wb",
        ) as f:
            pickle.dump(selected_data, f)
        selected_data = []

if cur_sentence_num != 0:
    selected_sentence_nums.append(cur_sentence_num)
    with open(os.path.join(save_dir, f"cross{len(selected_sentence_nums)}.pickle"), mode="wb") as f:
        pickle.dump(selected_data, f)

print(selected_sentence_nums)
