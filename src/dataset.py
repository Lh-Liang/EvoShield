import csv
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def parse_label(raw_label, label_map):
    if raw_label is None:
        return None

    normalized_label = str(raw_label).strip()
    if normalized_label == "":
        return None

    if normalized_label in label_map:
        return label_map[normalized_label]

    try:
        parsed_label = int(normalized_label)
    except (ValueError, TypeError):
        return None

    if parsed_label not in set(label_map.values()):
        return None

    return parsed_label

class PIDataset(Dataset):
    def __init__(self, data_path, model_path, pattern_id, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = self.tokenizer.get_vocab()
        vocab.pop(self.tokenizer.unk_token)
        vocab_list = sorted(vocab.keys())

        VERBALIZER = {
            "benign": vocab_list,
            "injection": vocab_list
        }
        VERBALIZER_INDEX_LABEL = {
            "benign": 0,
            "injection": 1
        }

        self.VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER.items()}

        self.examples = []
        self.mask = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max(len(v) for k, v in self.VERBALIZER_LABEL.items())
        self.m2c_tensor = self._build_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if len(row) < 3:
                    continue
                article_id, text, label = row[0], row[1], row[2]
                if idx == 0:
                    continue
                label = parse_label(label, VERBALIZER_INDEX_LABEL)
                if label is None:
                    continue
                example = [text, label]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def _build_m2c_tensor(self):
        m2c_tensor = torch.ones([len(self.VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1
        for label_idx, verbalizers in self.VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in self.VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids

    def encode(self, text):
        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', text]
        elif self.pattern_id == 1:
            prompt_text = [self.mask, 'type:', text]
        elif self.pattern_id == 2:
            prompt_text = [text, '(', self.mask, ')']
        elif self.pattern_id == 3:
            prompt_text = ['(', self.mask, ')', text]
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', text]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', text]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        return feature

    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        feature = self.encode(text)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        mlm_labels = self.get_mlm_labels(input_ids.tolist()[0])
        return input_ids, token_type_ids, attention_mask, mlm_labels, label, text


class JCDataset(Dataset):
    """
    Jailbreak Classification Dataset
    数据格式: prompt, type (benign/jailbreak)
    """
    def __init__(self, data_path, model_path, pattern_id, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = self.tokenizer.get_vocab()
        vocab.pop(self.tokenizer.unk_token)
        vocab_list = sorted(vocab.keys())

        VERBALIZER = {
            "benign": vocab_list,
            "jailbreak": vocab_list
        }
        VERBALIZER_INDEX_LABEL = {
            "benign": 0,
            "jailbreak": 1
        }

        self.VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER.items()}

        self.examples = []
        self.mask = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max(len(v) for k, v in self.VERBALIZER_LABEL.items())
        self.m2c_tensor = self._build_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                # CSV 格式: prompt, type
                if len(row) < 2:
                    continue
                prompt, label = row[0], row[1]
                if idx == 0:
                    # 跳过表头
                    if prompt.lower() == 'prompt' and label.lower() == 'type':
                        continue
                label = parse_label(label, VERBALIZER_INDEX_LABEL)
                if label is None:
                    continue
                example = [prompt, label]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def _build_m2c_tensor(self):
        m2c_tensor = torch.ones([len(self.VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1
        for label_idx, verbalizers in self.VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in self.VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids

    def encode(self, text):
        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', text]
        elif self.pattern_id == 1:
            prompt_text = [self.mask, 'type:', text]
        elif self.pattern_id == 2:
            prompt_text = [text, '(', self.mask, ')']
        elif self.pattern_id == 3:
            prompt_text = ['(', self.mask, ')', text]
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', text]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', text]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        return feature

    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        feature = self.encode(text)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        mlm_labels = self.get_mlm_labels(input_ids.tolist()[0])
        return input_ids, token_type_ids, attention_mask, mlm_labels, label, text


class SGDataset(Dataset):
    """
    SafeGuard Prompt Injection Dataset
    数据格式: ArticleId, text, label (0=benign, 1=injection)
    """
    def __init__(self, data_path, model_path, pattern_id, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = self.tokenizer.get_vocab()
        vocab.pop(self.tokenizer.unk_token)
        vocab_list = sorted(vocab.keys())

        VERBALIZER = {
            "benign": vocab_list,
            "injection": vocab_list
        }
        VERBALIZER_INDEX_LABEL = {
            "benign": 0,
            "injection": 1
        }

        self.VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER.items()}

        self.examples = []
        self.mask = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max(len(v) for k, v in self.VERBALIZER_LABEL.items())
        self.m2c_tensor = self._build_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                # CSV 格式: ArticleId, text, label
                if len(row) < 3:
                    continue
                article_id, text, label = row[0], row[1], row[2]
                if idx == 0:
                    # 跳过表头
                    if article_id.lower() == 'articleid' and text.lower() == 'text' and label.lower() == 'label':
                        continue
                label = parse_label(label, VERBALIZER_INDEX_LABEL)
                if label is None:
                    continue
                example = [text, label]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def _build_m2c_tensor(self):
        m2c_tensor = torch.ones([len(self.VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1
        for label_idx, verbalizers in self.VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in self.VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids

    def encode(self, text):
        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', text]
        elif self.pattern_id == 1:
            prompt_text = [self.mask, 'type:', text]
        elif self.pattern_id == 2:
            prompt_text = [text, '(', self.mask, ')']
        elif self.pattern_id == 3:
            prompt_text = ['(', self.mask, ')', text]
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', text]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', text]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        return feature

    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        feature = self.encode(text)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        mlm_labels = self.get_mlm_labels(input_ids.tolist()[0])
        return input_ids, token_type_ids, attention_mask, mlm_labels, label, text



def collate_fn(batch):
    input_ids, token_type_ids, attention_mask, mlm_labels, labels, texts = zip(*batch)
    input_ids = torch.stack([w.squeeze() for w in input_ids])
    token_type_ids = torch.stack([w.squeeze() for w in token_type_ids])
    attention_mask = torch.stack([w.squeeze() for w in attention_mask])
    mlm_labels = torch.stack([torch.Tensor(mlm_label).long() for mlm_label in mlm_labels])
    labels = torch.stack([torch.Tensor([label]).long() for label in labels])
    texts = list(texts)

    return input_ids, token_type_ids, attention_mask, mlm_labels, labels, texts
