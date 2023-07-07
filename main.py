from dataloader import HTRDataset
from transformers import TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, VisionEncoderDecoderModel
from datasets import load_metric


if __name__ == '__main__':
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=200,
        fp16=True,
        output_dir="/data/otueselm/ssda/",
        logging_steps=2,
        save_steps=1000,
        eval_steps=1000,

    )

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    max_target_length = 128

    train_dataset = HTRDataset(processor, max_target_length, split="train")
    eval_dataset = HTRDataset(processor, max_target_length, split="validation")

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    cer_metric = load_metric("cer")


    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        print(labels_ids[0], label_str[0], label_str[0])

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}


    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    # print(model.config.decoder.vocab)

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

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

