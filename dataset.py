import pandas as pd
import torch.utils.data as data

from PIL import Image
from transformers import RobertaTokenizer


class CaptionDataset(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.texts = None
        self.masks = None
        self.captions = self.read_captions()
        self.tokenize()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image = self.open_image(self.captions.filename[index])

        text = self.texts[index]
        mask = self.masks[index]

        return self.transform(image), text, mask
    
    def read_captions(self):
        return pd.read_csv(f'{self.root}/captions.csv')

    def tokenize(self):
        captions = []
        tokenizer = RobertaTokenizer.from_pretrained('pretrains/roberta', clean_up_tokenization_spaces=True)

        for caption in self.captions.caption:
            captions.append(caption)

        tokenized_outputs = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)

        self.texts = tokenized_outputs.input_ids
        self.masks = tokenized_outputs.attention_mask

    def open_image(self, filename):
        return Image.open(f'{self.root}/images/{filename}')
