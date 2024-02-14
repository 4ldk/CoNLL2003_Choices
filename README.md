# CoNLL2003 choices data

The json data in this repository was generated by 500 inferences in each sentence on CoNLL2003 train, validation, test data, CoNLL++ (2023), and CoNLL++ (CrossWeigh), using subword regularization (BPE-Dropout) with hyperparameter p = 0.1.
- For the training data, data was generated by five rounds of inferences for 20% of the train data by the model trained on the remaining 80%. (train_cross_var.py)
- For the other data, each data was generated by inferences by all train data. (train_all_train_data.py)
  - We used Roberta-base model fine-tuned by CoNLL2003 train dataset. You can use our model and see detail train config in [huggingface](https://huggingface.co/4ldk/Roberta-Base-CoNLL2003/blob/main/README.md).

All datasets can be found at `choice_data/`

We removed the original token and label data. Therefore, only the inference labels, the times they were predicted in 500, and the f1 score with the golden labels are available in this repository. If you have original CoNLL2003 data, you can add original token and label data to our datasets.
- F1 score is calculated by [seqeval](https://github.com/chakki-works/seqeval), but there are only O labels in both of predicted labels and golden labels, we set the score at 1.0.

## Setting
1. `git clone git@github.com:4ldk/CoNLL2003_Choices.git`
2. `cd CoNLL2003_Choices`
3. `mkdir row_data`
4. Copy each data to `row_data/`
    - Copy Original data (`eng.train`, `eng.testa`, `eng.tesb`) directly.
        - The following two repositories also have original data, but some data that are not suitable for training have been erased. The data and models published in this repository is based on data that has not been erased.
    - Copy `conllpp.txt` of [CoNLL++ (2023)](https://github.com/ShuhengL/acl2023_conllpp) directly.
    - Copy `conllpp_test.txt` of [CoNLL++ (CrossWeigh)](https://github.com/ZihanWangKi/CrossWeigh) as `conllcw.txt`
5. `pip install -r requirement.txt`

## Add original label and token information to choice_data
1. `python3 ./src/add_row_data.py`

## Create choice data by using the trained model
1. `mkdir model`
2. Edit `config/config2003.yaml` to set `test` test data, `loop` the number of test loops, `pred_p` subword regularization hyperparameter p, `load_local_model: False` and `test_model_name: "4ldk/Roberta-Base-CoNLL2003"`.
3. `python3 ./src/predictor.py`

## Train models and create train choice data
1. Edit `config/config2003.yaml`
2. If you want to change the number of divisions in the train data, Change line 12 of `make_cross_data.py` and lines 106 and 108 of `train_cross_var.py`.
3. `python3 ./src/make_cross_data.py`
4. `python3 ./src/train_cross_var.py`

## Train a model by all train data and create choice data
1. Edit `config/config2003.yaml`
    - Set `load_local_model: True` and others you want to change
2. `python3 ./src/train_all_train_data.py`
3. `python3 ./src/predictor.py`

## Reference
1. [BPE-Dropout: Simple and Effective Subword Regularization](https://aclanthology.org/2020.acl-main.170/)
2. [CoNLL2003](https://www.clips.uantwerpen.be/conll2003/ner/)
3. [Do CoNLL-2003 Named Entity Taggers Still Work Well in 2023?](https://aclanthology.org/2023.acl-long.459.pdf) ([github](https://github.com/ShuhengL/acl2023_conllpp))
4. [CrossWeigh: Training Named Entity Tagger from Imperfect Annotations](https://aclanthology.org/D19-1519/) ([github](https://github.com/ZihanWangKi/CrossWeigh))
