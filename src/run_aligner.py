import os
import sys
from tkinter.messagebox import NO 
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
import sklearn
import torch
import torch.nn.functional as F
import math
import random
import re
from transformers.hf_argparser import HfArgumentParser
from utils.aligner import *

class Trainer:
    def __init__(self, args, device):
        if device is None:
            self.device = 'cuda'
        else:
            self.device = device
        self.args = args
        if 'pkl' in self.args.model_name_or_path:
            self.model = torch.load(self.args.model_name_or_path)
            print(f'load best model from {self.args.model_name_or_path}')
        else:
            self.model = ModelDefine(args)
        self.model.to(self.device)
        self.train_loss = AverageMeter()
        self.updates = 0
        self.group_size = 8
        self.gap = 0.2
        self.optimizer = optim.Adamax([p for p in self.model.parameters() if p.requires_grad], lr=args.learning_rate)
        self.dt = DataManager(args)
        self.best_acc = 0.
    def train(self):
        for i in range(self.args.epochs):
            print("=" * 30)
            print("epoch%d" % i)
            with tqdm(enumerate(self.dt.iter_batches(which="train", batch_size=self.args.train_batch_size)), ncols=80) as t:
                for batch_id, batch in t:
                    self.model.train()
                    input_ids, token_type_ids, attention_mask, labels = [
                        Variable(e).long().to(self.device) for e
                        in batch]
                    labels = labels.clone().detach().unsqueeze(0)
                    _,loss = self.model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        do_train=True,) 
                    self.train_loss.update(loss.item(), self.args.train_batch_size)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    t.set_postfix(loss=self.train_loss.avg)
                    t.update(1)
                    self.updates += 1

            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset()
            acc = self.validate(epoch=i, which='dev')
            if acc > self.best_acc:
                self.best_acc = acc
                if not os.path.exists(self.args.model_save_dir):
                    os.mkdir(self.args.model_save_dir)
                save_path = os.path.join(self.args.model_save_dir,'checkpoint_best.pkl')
                print(f"save model to {save_path}")
                torch.save(self.model, save_path)

    def validate(self, which="test", epoch=-1):
        def rank_acc(labels, scores):
            labels = labels.reshape(-1,self.group_size)
            scores = scores.reshape(-1,self.group_size)
            acc = []
            for label, score in zip(labels, scores):
                if max(score) - score[0] >= self.gap:
                    top_index = np.argmax(score)
                else:
                    top_index = 0
                acc.append(label[top_index])
            return sum(acc)/len(acc)
        gold_label = []
        y_predprob = []
        for batch in tqdm(self.dt.iter_batches(which=which, batch_size=self.args.eval_batch_size), ncols=80):
            self.model.eval()
            input_ids, token_type_ids, attention_mask, labels = [
                    Variable(e).long().to(self.device) for e
                    in batch]

            labels = labels.clone().detach().unsqueeze(0)
            logits = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                do_train=False,)

            gold_label.extend(batch[-1])
            y_predprob.extend(F.softmax(logits, dim=1).detach().cpu().numpy()[:, -1])
        gold_label = np.array(gold_label)
        y_predprob = np.array(y_predprob)
        acc = rank_acc(gold_label, y_predprob)
        if acc >= self.best_acc and not self.args.do_train:
            predict_results = []
            align_acc = []
            before_align_acc = []
            with open(self.args.output_file) as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    item['align_scores'] = np.float64(y_predprob[idx*self.group_size:(idx+1)*self.group_size]).tolist()
                    if max(item['align_scores']) - item['align_scores'][0] >= self.gap:
                        top_index = np.argmax(item['align_scores'])
                    else:
                        top_index = 0
                    item['align_select'] = item["select_candidates"][top_index//len(item["structure_labels"])]
                    item['align_structure'] = item["structure_candidates"][top_index%len(item["structure_labels"])]
                    align_acc.append(item["select_labels"][top_index//len(item["structure_labels"])] and item["structure_labels"][top_index%len(item["structure_labels"])])
                    before_align_acc.append(item["select_labels"][0] and item["structure_labels"][0])
                    predict_results.append(item)
            print(f"before align accuracy = {sum(before_align_acc)/len(before_align_acc)}; after align accuracy = {sum(align_acc)/len(align_acc)}")
            with open(self.args.output_file, "w") as f:
                json.dump(
                    predict_results,
                    f,
                    indent=4,
                )

        print(f"{which} rank acc={acc}")
        return acc          

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    args: Arguments
    args = parser.parse_args_into_dataclasses()[0]
    setup_seed(0)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'gpu'
    trainer = Trainer(
        args=args, 
        device=device
    )
    if args.do_train:
        trainer.train()
    elif args.do_test:
        trainer.validate(epoch=0, which='test')