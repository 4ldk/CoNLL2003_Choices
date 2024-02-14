import os
import shutil
import sys

import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from bpe_dropout import RobertaTokenizerDropout
from utils import dataset_encode, f1_score, get_dataloader

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


class trainer:
    def __init__(
        self,
        model_name="bert-base-cased",
        lr=1e-5,
        batch_size=16,
        length=512,
        accum_iter=2,
        weight_decay=0.01,
        weight=False,
        num_warmup_steps=None,
        num_training_steps=None,
        use_scheduler=False,
        init_scale=4096,
        post_sentence_padding=False,
        add_sep_between_sentences=False,
        device="cuda",
    ):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(ner_dict)).to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)

        self.loss_func = nn.CrossEntropyLoss(weight=weight, ignore_index=ner_dict["PAD"])
        self.mv_loss_func = nn.KLDivLoss(reduction="batchmean")
        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
        if use_scheduler:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)

        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.length = length
        self.use_scheduler = use_scheduler
        self.accum_iter = accum_iter
        self.post_sentence_padding = post_sentence_padding
        self.add_sep_between_sentences = add_sep_between_sentences

    def forward(self, input, mask, type_ids=None, label=None):
        logits = self.model(input, mask, type_ids).logits
        pred = logits.squeeze(-1).argmax(-1)

        if label is not None:
            loss = self.loss_func(logits.view(-1, len(ner_dict)), label.view(-1))
            return logits, pred.to("cpu"), loss

        return logits, pred.to("cpu")

    def step(self, batch, batch_idx, batch_num, train=True):
        input, mask, type_ids, label = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
        )

        if train:
            with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                logits, pred, loss = self.forward(input, mask, type_ids, label)

            self.scaler.scale(loss / self.accum_iter).backward()
            if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == batch_num):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.optimizer.step()
                self.scaler.update()

                if self.use_scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

        else:
            with torch.no_grad():
                with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                    logits, pred, loss = self.forward(input, mask, type_ids, label)
            label = batch[-1]

        return logits, pred, loss, label.to("cpu")

    def epoch_loop(self, epoch, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        bar = tqdm(loader, leave=False, desc=f"Epoch{epoch} ")
        batch_loss, preds, labels = [], [], []
        final_loss, acc, f1 = 0, 0, 0

        for batch_idx, batch in enumerate(bar):
            logits, pred, loss, label = self.step(batch, batch_idx, batch_num=len(loader), train=train)
            bar.set_postfix(loss=loss.to("cpu").item())
            if train:
                batch_loss.append(loss.to("cpu").item())
            else:
                preds.append(pred)
                labels.append(label.to("cpu"))

        if train:
            final_loss = sum(batch_loss) / len(batch_loss)
            save_path = f"./model/epoch{epoch}.pth"
            save_model = self.model.to("cpu").state_dict()
            torch.save(save_model, save_path)
        else:
            acc, f1 = self.get_score(preds, labels)
        return final_loss, acc, f1

    def train(
        self,
        tokenizer,
        train_dataset,
        train_loader,
        num_epoch,
        valid_loader=None,
        test_loader=None,
        p=0,
    ):
        f1s = []
        losses = []
        tes_f1s = []
        for epoch in tqdm(range(num_epoch)):
            loss, _, _ = self.epoch_loop(epoch, train_loader, train=True)
            losses.append(loss)
            log_sentence = f"Epoch{epoch}: loss: {loss}"

            if valid_loader is not None:
                _, valid_acc, valid_f1 = self.epoch_loop(epoch, valid_loader, train=False)
                f1s.append(valid_f1)
                log_sentence += f", val_acc: {valid_acc:.4f}, val_f1: {valid_f1:.4f}"
            else:
                f1s.append(None)

            if test_loader is not None:
                _, _, test_f1 = self.epoch_loop(epoch, test_loader, train=False)
                tes_f1s.append(test_f1)
                log_sentence += f", tes_f1: {test_f1:.4f}"
            else:
                tes_f1s.append(None)

            tqdm.write(log_sentence)

            if epoch != num_epoch - 1 and p != 0:
                train_data = dataset_encode(
                    tokenizer,
                    train_dataset,
                    p=p,
                    padding=self.length,
                    return_tensor=True,
                    subword_label="PAD",
                    post_sentence_padding=self.post_sentence_padding,
                    add_sep_between_sentences=self.add_sep_between_sentences,
                )

                train_loader = get_dataloader(train_data, batch_size=self.batch_size, shuffle=True)

        with open("./train_valid_score.csv", "w") as f:
            min_loss = 100
            max_f1 = 0
            best_epoch = 0
            for i, (loss, f1, t_f1) in enumerate(zip(losses, f1s, tes_f1s)):
                f.write(f"{i}, {loss}, {f1}, {t_f1}\n")
                if f1 is not None and f1 > max_f1:
                    max_f1 = f1
                    best_epoch = i
                elif f1 is None or f1 == max_f1:
                    if loss < min_loss:
                        min_loss = loss
                        best_epoch = i

            print(f"best_epoch: {best_epoch}")
            shutil.copy(f"./model/epoch{best_epoch}.pth", "./model/best.pth")

    def get_score(self, preds, labels):
        preds = torch.concatenate(preds)
        labels = torch.concatenate(labels)

        pad = torch.logical_not((torch.ones_like(labels) * ner_dict["PAD"] == labels))
        data_num = ((labels == labels) * pad).sum()

        acc = ((preds == labels) * pad).sum().item() / data_num

        preds = preds.view(-1)
        labels = labels.view(-1)
        f1 = f1_score(labels, preds, skip=ner_dict["PAD"]).tolist()

        return acc, f1


def train(
    train_datasets,
    valid_dataset,
    batch_size,
    lr,
    num_epoch,
    length,
    p,
    model_name,
    accum_iter=4,
    weight_decay=0,
    use_loss_weight=False,
    use_scheduler=False,
    warmup_late=0.01,
    post_sentence_padding=False,
    add_sep_between_sentences=False,
):
    device = "cuda"
    init_scale = 4096
    os.makedirs("./model", exist_ok=True)

    tokenizer = RobertaTokenizerDropout.from_pretrained(model_name, alpha=p)

    train_data = dataset_encode(
        tokenizer,
        train_datasets,
        p=p,
        padding=length,
        subword_label="PAD",
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
    )
    train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)

    valid_data = dataset_encode(
        tokenizer,
        valid_dataset,
        p=0,
        padding=length,
        subword_label="PAD",
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
    )
    valid_loader = get_dataloader(valid_data, batch_size=batch_size, shuffle=False, drop_last=False)

    weight = train_data["weight"].to(device) if use_loss_weight else None
    num_training_steps = int(len(train_loader) / accum_iter) * num_epoch
    num_warmup_steps = int(num_training_steps * warmup_late)

    net = trainer(
        model_name=model_name,
        lr=lr,
        batch_size=batch_size,
        length=length,
        accum_iter=accum_iter,
        weight_decay=weight_decay,
        weight=weight,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        use_scheduler=use_scheduler,
        init_scale=init_scale,
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
        device=device,
    )
    net.train(
        tokenizer,
        train_datasets,
        train_loader,
        num_epoch,
        valid_loader=valid_loader,
        p=p,
    )
