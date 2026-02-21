import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import os
import swanlab

os.environ["SWANLAB_PROJECT"]="qwen3-sft-bangumi-new"
PROMPT = "你是一个精通 ACG 文化的影评分析专家。你的任务是分析用户的 Bangumi 评论中表达出对于剧情、作画、音乐、角色维度的情感倾向及总体评价，并将其转化为符合以下定义的 JSON 格式。\n\n 输出要求:\n1. 必须严格遵守 JSON 语法，输出原始、紧凑的 JSON，不要包含任何空格或缩进，不要输出 markdown 代码块（```json）。\n2. JSON 必须包含以下字段:\n - \"aspects\": 对象。包含 \"story\", \"visual\", \"music\",\"character\" 四个键，值为 \"Positive\", \"Negative\", \"Neutral\" 或 \"Not Mentioned\"。\n - \"final_sentiment\": 字符串。整体评价，仅限 \"Positive\", \"Neutral\", \"Negative\"。\n - \"confidence\": 整数。1-10，表示你对该判断的置信度。\n\n 注意：不要编造用户未提到的观点。\n\n"
MAX_LENGTH = 2048

swanlab.config.update({
    "model": "Qwen/Qwen3-8B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    })

def process_func(example):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}  

def predict(messages, model, tokenizer):
    device = "cuda"

    model.eval()  # 关键：切换到评估模式
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    
    model.train()  # 切回训练模式
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# Transformers加载模型权重
# 在下面填入原模型保存路径
tokenizer = AutoTokenizer.from_pretrained(" ", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(" ", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 配置lora
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

# 加载、处理数据集和测试集
train_dataset_path = "./train_dataset.jsonl"
test_dataset_path = "./test_dataset.jsonl"

# 得到训练集
train_df = pd.read_json(train_dataset_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 得到验证集
eval_df = pd.read_json(test_dataset_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

print(train_dataset[0])

args = TrainingArguments(
    output_dir=" ", # 在这里填入模型保存路径
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-8B-nocot",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
trainer.save_state()
trainer.save_model(output_dir=args.output_dir)

# 用测试集的前3条，主观看模型
test_df = pd.read_json(test_dataset_path, lines=True)[:3]

test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()