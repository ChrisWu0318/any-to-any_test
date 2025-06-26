import argparse
import functools
import os
import platform

from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="dataset/train.json",       help="Path to the training dataset")
add_arg("test_data",     type=str, default="dataset/test.json",        help="Path to the testing dataset")
add_arg("base_model",    type=str, default="openai/whisper-tiny",      help="Base Whisper model")
add_arg("output_dir",    type=str, default="output/",                  help="Directory to save trained models")
add_arg("warmup_steps",  type=int, default=50,      help="Number of warm-up steps")
add_arg("logging_steps", type=int, default=100,     help="Steps between logging")
add_arg("eval_steps",    type=int, default=1000,    help="Steps between evaluations")
add_arg("save_steps",    type=int, default=1000,    help="Steps between checkpoints")
add_arg("num_workers",   type=int, default=8,       help="Number of data loading workers")
add_arg("learning_rate", type=float, default=1e-3,  help="Learning rate")
add_arg("min_audio_len", type=float, default=0.5,   help="Minimum audio length in seconds")
add_arg("max_audio_len", type=float, default=30,    help="Maximum audio length in seconds, cannot exceed 30")
add_arg("use_adalora",   type=bool,  default=True,  help="Whether to use AdaLoRA instead of LoRA")
add_arg("fp16",          type=bool,  default=True,  help="Enable fp16 training")
add_arg("use_8bit",      type=bool,  default=False, help="Quantize model to 8-bit")
add_arg("timestamps",    type=bool,  default=False, help="Use timestamps during training")
add_arg("use_compile",   type=bool, default=False, help="Use PyTorch 2.0 compiler")
add_arg("local_files_only", type=bool, default=False, help="Load models only from local cache")
add_arg("num_train_epochs", type=int, default=3,      help="Number of training epochs")
add_arg("language", type=str, default="Chinese", help="Language setting, full name or abbreviation. None for multilingual.")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="Model task")
add_arg("augment_config_path",         type=str, default=None, help="Path to data augmentation config file")
add_arg("resume_from_checkpoint",      type=str, default=None, help="Path to checkpoint for resuming training")
add_arg("per_device_train_batch_size", type=int, default=8,    help="Training batch size per device")
add_arg("per_device_eval_batch_size",  type=int, default=8,    help="Evaluation batch size per device")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="Gradient accumulation steps")
add_arg("push_to_hub",                 type=bool, default=False, help="Push model weights to HuggingFace Hub")
add_arg("hub_model_id",                type=str,  default=None,  help="Model repo ID on HuggingFace Hub")
add_arg("save_total_limit",            type=int,  default=10,  help="Limit number of checkpoints saved")
args = parser.parse_args()
print_arguments(args)

# Set num_workers to 0 on Windows
if platform.system() == "Windows":
    args.num_workers = 0

def main():
    # Load Whisper processor (feature extractor + tokenizer)
    processor = WhisperProcessor.from_pretrained(args.base_model,
                                                 language=args.language,
                                                 task=args.task,
                                                 no_timestamps=not args.timestamps,
                                                 local_files_only=args.local_files_only)

    # Load training and testing datasets
    train_dataset = CustomDataset(data_list_path=args.train_data,
                                  processor=processor,
                                  language=args.language,
                                  timestamps=args.timestamps,
                                  min_duration=args.min_audio_len,
                                  max_duration=args.max_audio_len,
                                  augment_config_path=args.augment_config_path)
    test_dataset = CustomDataset(data_list_path=args.test_data,
                                 processor=processor,
                                 language=args.language,
                                 timestamps=args.timestamps,
                                 min_duration=args.min_audio_len,
                                 max_duration=args.max_audio_len)
    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Load Whisper model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                            load_in_8bit=args.use_8bit,
                                                            device_map=device_map,
                                                            local_files_only=args.local_files_only)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = prepare_model_for_kbit_training(model)
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    print('Loading LoRA modules...')
    if args.resume_from_checkpoint:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        print('Adding LoRA modules...')
        target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
        print(target_modules)
        if args.use_adalora:
            total_step = args.num_train_epochs * len(train_dataset)
            config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                                   lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules,
                                   total_step=total_step)
        else:
            config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none")
        model = get_peft_model(model, config)

    if args.base_model.endswith("/"):
        args.base_model = args.base_model[:-1]
    output_dir = str(os.path.join(args.output_dir, os.path.basename(args.base_model)))

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(output_dir=output_dir,
                                             per_device_train_batch_size=args.per_device_train_batch_size,
                                             per_device_eval_batch_size=args.per_device_eval_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             warmup_steps=args.warmup_steps,
                                             num_train_epochs=args.num_train_epochs,
                                             save_strategy="steps",
                                             eval_strategy="steps",
                                             load_best_model_at_end=True,
                                             fp16=args.fp16,
                                             report_to=["tensorboard"],
                                             save_steps=args.save_steps,
                                             eval_steps=args.eval_steps,
                                             torch_compile=args.use_compile,
                                             save_total_limit=args.save_total_limit,
                                             optim='adamw_torch',
                                             ddp_find_unused_parameters=False if ddp else None,
                                             dataloader_num_workers=args.num_workers,
                                             logging_steps=args.logging_steps,
                                             remove_unused_columns=False,
                                             label_names=["labels"],
                                             push_to_hub=args.push_to_hub)

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print('=' * 90)
        model.print_trainable_parameters()
        print('=' * 90)

    # Define the trainer
    trainer = Seq2SeqTrainer(args=training_args,
                             model=model,
                             train_dataset=train_dataset,
                             eval_dataset=test_dataset,
                             data_collator=data_collator,
                             processing_class=processor.feature_extractor,
                             callbacks=[SavePeftModelCallback])
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint

    # Start training
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model state
    trainer.save_state()
    model.config.use_cache = True
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))

    # Push to HuggingFace Hub if needed
    if training_args.push_to_hub:
        hub_model_id = args.hub_model_id if args.hub_model_id is not None else output_dir
        model.push_to_hub(hub_model_id)

if __name__ == '__main__':
    main()