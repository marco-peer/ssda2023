import csv

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data as d
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def read_tsv(tsv='test/files.tsv'):
    with open(tsv) as file:
        tsv_file = list(csv.reader(file, delimiter="\t"))
    return tsv_file


def write_tsv(results, tsv):
    with open(tsv, 'wt') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        for el in results:
            tsv_writer.writerow(el)




class HTRDataset(d.Dataset):

    def __init__(self, src, pre_processor):
        self.data = src
        self.pre_processor = pre_processor

    def __getitem__(self, index):
        (image_path, transcription) = self.data[index]
        image = Image.open("%s/%s" %("/data/otueselm/ssda/data/test",image_path)).convert("RGB")
        pixel_values = self.pre_processor(image, return_tensors="pt").pixel_values.squeeze()

        return pixel_values, image_path

    def __len__(self):
        return len(self.data)


def test_model(model, processor, src_tsv='test/gt.tsv', out='test/result.tsv'):
    src = read_tsv(src_tsv)

    test_data = HTRDataset(src, processor)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

    transcriptions = []
    for (images, image_paths) in tqdm(test_dataloader):
        images = images.cuda()
        #print(images.shape)
        generated_ids = model.generate(images)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for image_path, gen_text in zip(image_paths, generated_text):
            transcriptions.append([image_path,gen_text])

    write_tsv(transcriptions, tsv=out)


if __name__ == '__main__':
    output_file = "/data/otueselm/ssda/checkpoint-212000/"
    model = VisionEncoderDecoderModel.from_pretrained(output_file).cuda()

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4


    test_model(model, processor, src_tsv='/data/otueselm/ssda/data/test/gt.tsv', out='/data/otueselm/ssda/data/test/result_final.tsv')