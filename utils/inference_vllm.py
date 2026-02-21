import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.sampling_params import GuidedDecodingParams # vllm版本大于12.0可以使用vllm.sampling_params
import streamlit as st

# 配置路径和参数

model_path_CoT = "/model/Qwen-awq-4bit"
model_path_noCoT = "/model/Qwen-merged-nocot-awq"

# json schema 定义（分析模式 vs 非分析模式）

schema_CoT = {
  "type": "object",
  "properties": {
    "glossary": {
      "type": "object",
      "additionalProperties": { "type": "string" },
    },
    "step_by_step_analysis": { "type": "string" },
    "aspects": {
      "type": "object",
      "properties": {
        "story": { "type": "string" },
        "visual": { "type": "string" },
        "music": { "type": "string" },
        "character": { "type": "string" }
      },
      "required": ["story", "visual", "music", "character"]
    },
    "final_sentiment": { "type": "string" },
    "confidence": { "type": "integer", "minimum": 0, "maximum": 10 }
  },
  "required": ["glossary", "step_by_step_analysis", "aspects", "final_sentiment", "confidence"]
}

schema_noCoT = {
  "type": "object",
  "properties": {
    "aspects": {
      "type": "object",
      "properties": {
        "story": { "type": "string" },
        "visual": { "type": "string" },
        "music": { "type": "string" },
        "character": { "type": "string" }
      },
      "required": ["story", "visual", "music", "character"]
    },
    "final_sentiment": { "type": "string" },
    "confidence": { "type": "integer", "minimum": 0, "maximum": 10 }
  },
  "required": ["aspects", "final_sentiment", "confidence"]
}

@st.cache_resource
def load_vllm_model(path):
    return LLM(
        model=path,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        dtype="bfloat16"
    )

# 初始化 vLLM 推理引擎
def inference_with_vllm(input_file, output_file, instruction, st_bar, with_CoT=None, batch_size=20):
    schema = schema_CoT if with_CoT == "分析模式" else schema_noCoT
    model_path = model_path_CoT if with_CoT == "分析模式" else model_path_noCoT
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = load_vllm_model(model_path)

    guided_decoding_params = GuidedDecodingParams(
        json=schema  # 直接传入 JSON schema
    )


    # temperature: 采样温度
    # top_p: 核采样
    # max_tokens: 最大生成长度
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        guided_decoding=guided_decoding_params
    )

    # 读取数据并准备 Prompt 列表
    raw_data = []
    prompts = []

    print("正在预处理数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_data.append(item)
            input_value = item.get('text', "")
            # instruction = item.get('instruction', PROMPT)
            
            # 构造对话格式并应用模板
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_value}
            ]
            # apply_chat_template 将对话转为模型能理解的字符串
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(prompt)

    total_count = len(prompts)
    all_results = []

    # --- 开始分批次推理 ---
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(0, total_count, batch_size):
            # 获取当前批次
            batch_prompts = prompts[i : i + batch_size]
            batch_raw = raw_data[i : i + batch_size]
            
            # 推理当前批次
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            
            # 处理结果
            for j, output in enumerate(batch_outputs):
                generated_text = output.outputs[0].text
                result_item = {
                    "instruction": instruction,
                    "input": batch_raw[j].get('text', ""),
                    "output": generated_text
                }
                all_results.append(result_item)
                f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            
            # --- 更新进度条 ---
            processed_count = min(i + batch_size, total_count)
            progress_pct = processed_count / total_count
            st_bar.progress(progress_pct, text=f"本地模型推理中... 进度: {processed_count}/{total_count}")

    return all_results