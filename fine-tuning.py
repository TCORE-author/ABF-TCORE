import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_dataset, DatasetDict
import transformers
import trl

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="project_name")
    wandb_entity: Optional[str] = field(default=None)
    train_file_path: Optional[str] = field(default='path/to/tokenized-dataset')
    dagger: bool = field(default=False)
    num_curriculum_stages: int = field(default=3)

    def __post_init__(self):
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity


def split_dataset_into_curriculum_stages(
    dataset_dict: DatasetDict,
    num_stages: int = 3,
    trajectory_field: str = "deepseek_thinking_trajectory",
    text_field: str = "text"
) -> list:
    def get_difficulty(example):
        return len(example[trajectory_field])

    train_dataset = dataset_dict['train'].map(lambda x: {"difficulty": get_difficulty(x)})
    train_dataset = train_dataset.sort("difficulty")

    total_len = len(train_dataset)
    stage_size = total_len // num_stages
    stages = []

    for i in range(num_stages):
        start_idx = i * stage_size
        end_idx = (i + 1) * stage_size if i < num_stages - 1 else total_len
        stage_train = train_dataset.select(range(start_idx, end_idx))
        stage_dataset_dict = DatasetDict({"train": stage_train})

        for split_name in dataset_dict.keys():
            if split_name != "train":
                stage_dataset_dict[split_name] = dataset_dict[split_name]

        stages.append(stage_dataset_dict)

    return stages


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    kwargs = {}
    if "70B" in config.model_name:
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    dataset = load_dataset(config.train_file_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"

    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size

    curriculum_stages = split_dataset_into_curriculum_stages(
        dataset_dict=dataset,
        num_stages=config.num_curriculum_stages,
        trajectory_field="deepseek_thinking_trajectory",
        text_field=args.dataset_text_field
    )

    for stage_idx, stage_dataset in enumerate(curriculum_stages):
        logging.info(
            f"Starting curriculum stage {stage_idx + 1}/{config.num_curriculum_stages} "
            f"with {len(stage_dataset['train'])} training examples."
        )

        trainer = trl.SFTTrainer(
            model=model,
            train_dataset=stage_dataset['train'],
            eval_dataset=(
                stage_dataset['test']
                if 'test' in stage_dataset
                else stage_dataset['train']
            ),
            args=args,
            data_collator=collator
        )

        trainer.train()
        stage_output_dir = os.path.join(args.output_dir, f"stage_{stage_idx + 1}")
        trainer.save_model(output_dir=stage_output_dir)
        tokenizer.save_pretrained(stage_output_dir)

    trainer.accelerator.wait_for_everyone()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Training completed across all curriculum stages.")


if __name__ == "__main__":
    train()
