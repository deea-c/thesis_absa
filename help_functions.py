import pandas as pd
import xml.etree.ElementTree as ET
def read_xml_files(file_path ):
  tree = ET.parse(file_path)
  root = tree.getroot()

  reviews = []
  sentences = []

  for review in root.findall('Review'):
      rid = review.get('rid')
      for sentence in review.findall('sentences/sentence'):
          sentence_id = sentence.get('id')
          sentence_text =  sentence.find('text').text
          opinions = sentence.find('Opinions')
          sentence_opinions = []
          if opinions is not None:
              for opinion in opinions.iter('Opinion'):
                  opinion_target = opinion.get('target')
                  opinion_category = opinion.get('category')
                  opinion_polarity = opinion.get('polarity')
                  from_ = int(opinion.get('from'))
                  to_ = int(opinion.get('to'))
                  sentence_opinions.append((opinion_target, opinion_category, opinion_polarity, from_ , to_))
          sentences.append((sentence_id, sentence_text, sentence_opinions))
      reviews.append(sentences)
      sentences = []

  df_rows = []
  for review in reviews:
      for sentence in review:
          sentence_id, sentence_text, sentence_opinions = sentence
          for opinion in sentence_opinions:
              opinion_target, opinion_category, opinion_polarity  , from_ , to_= opinion
              df_rows.append((review[0][0], sentence_id, sentence_text, opinion_target, opinion_category, opinion_polarity, from_ , to_))
              
  df = pd.DataFrame(df_rows, columns=['Review_id', 'Sentence_id', 'Text', 'Opinion_target', 'Opinion_category', 'Opinion_polarity', "From", "To"])
  return df

# def get_annotations_as_dict(df):
#         annotations = []
#         for _, row in df.iterrows():
#             annotation = {
#                 'Opinion_target': row['Opinion_target'],
#                 'Opinion_category': row['Opinion_category'],
#                 'Opinion_polarity': row['Opinion_polarity'],
#                 'From': row['From'],
#                 'To': row['To']
#             }
#             annotations.append(annotation)

#         df1 = pd.DataFrame({
#             'Sentence_id': df['Sentence_id'],
#             'Text': df['Text'],
#             'annotations': annotations
#         })
#         grouped = df1.groupby(['Sentence_id','Text'])[["annotations"]].apply(lambda x: list(x["annotations"])).reset_index(name="annotations")

#         return grouped

def get_annotations_as_dict(df):
    annotations = []
    polarities = []
    
    for _, row in df.iterrows():
        annotation = {
            'Opinion_target': row['Opinion_target'],
            'Opinion_category': row['Opinion_category'],
            'Opinion_polarity': row['Opinion_polarity'],
            'From': row['From'],
            'To': row['To']
        }
        annotations.append(annotation)
        polarities.append(row['Opinion_polarity'])
    
    df1 = pd.DataFrame({
        'Sentence_id': df['Sentence_id'],
        'Text': df['Text'],
        'annotations': annotations,
        'polarities': polarities
    })
    
    grouped = df1.groupby(['Sentence_id', 'Text']).agg({'annotations': list, 'polarities': ','.join}).reset_index()

    return grouped
    
# pip install evaluate
#pip install seqeval (need to install this)

# from help_functions import get_annotations_as_dict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from typing_extensions import TypedDict
from typing import List,Any
IntList = List[int] # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch
from transformers import AutoTokenizer,  BatchEncoding
from tokenizers import Encoding
import transformers


# https://github.com/LightTag/sequence-labeling-with-transformers/blob/master/notebooks/how-to-align-notebook.ipynb procedure followed from here ( changed to BIO as BILOU is not supported in the pipeline)
def align_tokens_and_annotations_bio(tokenized: Encoding, annotations):
  tokens = tokenized.tokens
  aligned_labels = ["O"] * len(tokens)
  for anno in annotations:
      annotation_token_ix_set = set()
      for char_ix in range(anno["From"], anno["To"]):
          token_ix = tokenized.char_to_token(char_ix)
          if token_ix is not None:
              annotation_token_ix_set.add(token_ix)
      if len(annotation_token_ix_set) == 1:
          token_ix = annotation_token_ix_set.pop()
          prefix = "B"
          aligned_labels[token_ix] = f"{prefix}-{anno['Opinion_polarity']}"
      else:
          sorted_token_ixs = sorted(annotation_token_ix_set)
          for i, token_ix in enumerate(sorted_token_ixs):
              if i == 0:
                  prefix = "B"
              else:
                  prefix = "I"
              aligned_labels[token_ix] = f"{prefix}-{anno['Opinion_polarity']}"
  return aligned_labels

import itertools


class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremntal ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BI")):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l
        # Add the OUTSIDE label - no label for the token

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
        raw_labels = align_tokens_and_annotations_bio(tokenized_text, annotations)
        return list(map(self.labels_to_id.get, raw_labels))


# https://github.com/LightTag/sequence-labeling-with-transformers/blob/master/notebooks/how-to-align-notebook.ipynb function taken from this link

from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import AutoTokenizer

@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    labels: IntList


class TraingDataset(Dataset):
    def __init__(
        self,
        data: Any,
        label_set: LabelSet,
        tokenizer: AutoTokenizer,
        tokens_per_batch=32,
        window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        self.texts = []
        self.annotations = []

        for i, example in data.iterrows():
            self.texts.append(example["Text"])
            self.annotations.append(example["annotations"])
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )
            aligned_labels.append(aligned)
        ###END OF LABEL ALIGNMENT

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        empty_label_id = "O"
        for encoding, label in zip(tokenized_batch.encodings, aligned_labels):
            length = len(label)  # How long is this sequence
            for start in range(0, length, self.window_stride):

                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                self.training_examples.append(
                    TrainingExample(
                        # Record the tokens
                        # The ids of the tokens
                        input_ids=encoding.ids[start:end]
                        + [self.tokenizer.pad_token_id]
                        * padding_to_add,  # padding if needed
                        labels=(label[start:end] + [-100] * \
                                padding_to_add),  # padding if needed \
                        # -100 is a special token for padding of labels,
                        attention_masks=(
                            encoding.attention_mask[start:end]
                            + [0]
                            * padding_to_add  # 0'd attenetion masks where we added padding
                        ),
                    )
                )

    def __len__(self):
        return len(self.training_examples)
    def __getitem__(self, idx) -> dict:
        training_example = self.training_examples[idx]
        return training_example.__dict__
    


import evaluate
import numpy as np
metric = evaluate.load("seqeval",mode='strict', scheme="IOB2")

label_set = LabelSet(labels=["positive", "negative", "neutral"])
label_list = label_set.ids_to_label

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # print("predictions:", predictions)
    # print("labels:", labels)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    
    results = metric.compute(predictions=true_predictions, references=true_labels)
 
    return {
        "val_precision": results["overall_precision"],
        "val_recall": results["overall_recall"],
        "val_f1": results["overall_f1"],
        "val_accuracy": results["overall_accuracy"],
    }