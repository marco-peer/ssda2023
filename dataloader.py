import torch
import torch.utils.data as d
from PIL import Image


# <image-path>\t<transcription>`


def load_dataset(split="train"):
    data_path = "/data/otueselm/ssda/data/%s/gt.tsv" % split
    data_lines = open(data_path, "r").readlines()
    data = []
    for d in data_lines:
        image_id, transcription = d.rstrip().split("\t")
        data.append(("/data/otueselm/ssda/data/%s/%s" % (split,image_id),transcription))

    return data


class HTRDataset(d.Dataset):

    def __init__(self, pre_processor, max_target_length, split="train"):
        self.root_dir = "/data/otueselm/ssda/data/%s/" % split
        self.pre_processor = pre_processor
        self.data = load_dataset(split)
        self.max_target_length = max_target_length

    def __getitem__(self, index):
        (image_id, transcription) = self.data[index]
        image = Image.open(image_id).convert("RGB")
        pixel_values = self.pre_processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.pre_processor.tokenizer(transcription,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.pre_processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

        return encoding

    def __len__(self):
        return len(self.data)