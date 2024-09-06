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
        self.captions = pd.read_csv(f'{root}/captions.csv')
        self.tokenize()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        filename = self.captions.filename[index]

        image = self.open(filename)
        image = self.transform(image)

        return image, self.texts[index], self.masks[index]

    def tokenize(self):
        captions = []
        tokenizer = RobertaTokenizer.from_pretrained('weights/develop/roberta', clean_up_tokenization_spaces=True)

        for caption in self.captions.caption:
            captions.append(caption)

        tokenized_outputs = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)

        self.texts = tokenized_outputs.input_ids
        self.masks = tokenized_outputs.attention_mask

    def open(self, filename):
        return Image.open(f'{self.root}/images/{filename}')
