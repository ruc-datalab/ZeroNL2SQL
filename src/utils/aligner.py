from transformers import DebertaV2Model, DebertaV2Tokenizer
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from packaging import version
import os
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import torch
import json
from typing import Optional
from dataclasses import dataclass, field
import random
from collections import Counter
@dataclass
class Arguments:
    """
    Arguments for training model.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenizer."},
    )
    model_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the finetuned model."},
    )
    learning_rate: float = field(
        default=None,
        metadata={"help": ""}
    )
    train_batch_size: int = field(
        default=None,
        metadata={"help": ""}
    )
    eval_batch_size: int = field(
        default=None,
        metadata={"help": ""}
    )
    epochs: int = field(
        default=None,
        metadata={"help": ""}
    )
    do_train: bool = field(
        default=None,
        metadata={"help": ""}
    )
    do_test: bool = field(
        default=None,
        metadata={"help": ""}
    )
    load_checkpoints: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load the fine-tuned checkpoint."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the train set"},
    )
    dev_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dev set"},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test set"},
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save result"},
    )
    max_train_samples: int = field(
        default=None,
        metadata={"help": "max train samples"}
    )

class DataManager:
    def __init__(self, args):
        #load tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer_path)
        self.tokenizer.model_max_length = 512
        self.args = args
        self.prepare_data()

    def prepare_data(self):
        if self.args.do_train:
            train_smps = self.get_data_2s('train')
            dev_smps = self.get_data_2s('dev')
            self.train_dataset = MyDataSet(train_smps)
            self.dev_dataset = MyDataSet(dev_smps)
        if self.args.do_test:

            test_smps = self.get_data_2s('test')
            self.test_dataset = MyDataSet(test_smps)

    def get_data_2s(self, which):
        data = []
        if which == 'train':
            data = json.load(open(self.args.train_file))
        elif which == 'dev':
            data = json.load(open(self.args.dev_file))
        elif which == 'test':
            fnm = self.args.test_file
            with open(fnm) as f:
                hypotheses_data = json.load(f)
                for idx, item in enumerate(hypotheses_data):
                    if idx%3 == 1:
                        question = item["input"][85:item["input"].index(', database: Database: ')]
                        label = dict(Counter(item["real_label"][7:].split(', ')))
                        sample_strs = []
                        labels = []
                        gold_select = item["real_label"]
                        for hyp in item["topk_preds"][:4]:
                            sample_strs.append(hyp)
                            hyp_col = dict(Counter(hyp[7:].split(', ')))
                            if hyp_col == label:
                                labels.append(1.)
                            else:
                                labels.append(0.)

                        structures = hypotheses_data[idx-1]["topk_preds"]
                        gold_structure = hypotheses_data[idx-1]["real_label"]
                        structure_candidates = []
                        structure_labels = []
                        for pred_structure in structures:
                            keywords = set(pred_structure.split('-'))
                            if keywords <= {'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'INTERSECT', 'UNION', 'EXCEPT'}:
                                structure_candidates.append(pred_structure)
                                if pred_structure == gold_structure:
                                    structure_labels.append(1.)
                                else:
                                    structure_labels.append(0.)
                            if len(structure_candidates) == 2:
                                break
                        if len(structure_candidates) != 2: 
                            structure_candidates.append(structure_candidates[0])
                            structure_labels.append(structure_labels[0])
                        tables = hypotheses_data[idx+1]["topk_preds"]
                        gold_table = hypotheses_data[idx+1]["real_label"]
                        data.append({
                            "question": question,
                            "gold_select": gold_select,
                            "select_candidates": sample_strs,
                            "select_labels": labels,
                            "gold_structure": gold_structure,
                            "structure_candidates": structure_candidates,
                            "structure_labels": structure_labels,
                            "gold_table": gold_table,
                            "table_candidates": [x for x in tables if x != ""],
                        })
        with open(self.args.output_file, 'w') as f:
            json.dump(data, f, indent=4)
        smps = []
        input_str = "user question: {question} | our solution: {select_clause}"
        for idx,item in enumerate(tqdm(data)):
            if which == "train" and self.args.max_train_samples and idx > self.args.max_train_samples:
                break
            new_smps = {"pos": [], "neg": []}
            if (not (0. in item["select_labels"] and 1. in item["select_labels"]) or 1. not in item["structure_labels"]) and which == "train":
                continue
            for select_str, select_label in zip(item["select_candidates"], item["select_labels"]):
                for structure_str, structure_label in zip(item["structure_candidates"], item["structure_labels"]):
                    pair = select_str + ' ' + ' … '.join(structure_str.split('-')[1:])
                    if idx < 2:
                        print(input_str.format(question=item["question"], select_clause=pair).lower())
                    input = self.tokenizer(input_str.format(question=item["question"], select_clause=pair).lower(), padding="max_length", truncation=True, return_tensors="pt")
                    label = select_label and structure_label
                    if which != "train":
                        smps.append([input["input_ids"][0],input["token_type_ids"][0],input["attention_mask"][0],label])
                    elif label:
                        new_smps["pos"].append([input["input_ids"][0],input["token_type_ids"][0],input["attention_mask"][0],label])
                    else:
                        new_smps["neg"].append([input["input_ids"][0],input["token_type_ids"][0],input["attention_mask"][0],label])
            if which == "train": # 1 batch: 1 pos + 3 neg
                for pos_smp in new_smps["pos"]:
                    smp_group = [pos_smp]+random.sample(new_smps["neg"], 3)
                    smps.extend(smp_group)
        if len(smps)%4 != 0 and which == 'train':
            smps = smps[:-2]
        print(f'{which} sample num={len(smps)}')
        return smps


    def iter_batches(self, which="train", samples=None, batch_size=None):
        if which == 'train':
            return DataLoader(shuffle=False, dataset=self.train_dataset, batch_size=batch_size)
        elif which == 'dev':
            return DataLoader(shuffle=False, dataset=self.dev_dataset, batch_size=batch_size)
        elif which == 'test':
            return DataLoader(shuffle=False, dataset=self.test_dataset, batch_size=batch_size)


class MyDataSet(Dataset):
    def __init__(self, smps):
        self.smps = smps
        super().__init__()

    def __getitem__(self, i):
        input_ids, token_type_ids, attention_mask, label= self.smps[i]
        return input_ids, token_type_ids, attention_mask, label
    def __len__(self):
        return len(self.smps)

def linear_act(x):
    return x

def _mish_python(x):
    return x * torch.tanh(nn.functional.softplus(x))

if version.parse(torch.__version__) < version.parse("1.9"):
    mish = _mish_python
else:
    mish = nn.functional.mish

def gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

def _silu_python(x):
    return x * torch.sigmoid(x)


if version.parse(torch.__version__) < version.parse("1.7"):
    silu = _silu_python
else:
    silu = nn.functional.silu

def gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = gelu_python
else:
    gelu = nn.functional.gelu

ACT2FN = {
    "relu": nn.functional.relu,
    "silu": silu,
    "swish": silu,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_python": gelu_python,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
}

class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True

def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout
class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None

# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training
    Args:
        drop_prob (float): the dropout probabilities
    """
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob

class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    def output_dim(self):
        return self.config.hidden_size
    

class ModelDefine(nn.Module):
    def __init__(self, args):
        super(ModelDefine, self).__init__()
        print(args.model_name_or_path)
        self.deberta = DebertaV2Model.from_pretrained(args.model_name_or_path)
        self.config = self.deberta.config
        num_labels = getattr(self.config, "num_labels", 2)
        self.num_labels = num_labels
        self.config.num_labels = num_labels
        self.config.classifier_dropout = 0.0
        self.pooler = ContextPooler(self.config)
        self.classifier = nn.Linear(1024,2)
        drop_out = getattr(self.config,"cls_dropout",None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.alpha = 0.5
    
    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        do_train=None,
    ):       
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if do_train:
            loss = None
            BPR_loss = BPRLoss()
            CE_loss = CrossEntropyLoss()
            loss = CE_loss(logits.view(-1, self.num_labels), labels.view(-1))
            return logits,loss
        else:
            return logits