
import csv

from PIL import Image
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
        
def load_img(imgp):
    img = Image.open(imgp)
    return imgp

def test_model(model, src_tsv='test/files.tsv', out='test/result.tsv'):
    src = read_tsv(src_tsv)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    for el in src:
        img = load_img(el[0])
        
        ###
        pixel_values = processor(load_image(img), return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ###

        el[1] = generated_text

    write_tsv(src, tsv=out)

class DummyModel:

    def __init__(self):
        pass
    def __call__(self, img):
        return 's'

if __name__ == '__main__':
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    test_model(model, src_tsv='test/files.tsv', out='test/result.tsv')
        
