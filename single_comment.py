import streamlit as st
import json
import plotly.graph_objects as go
import json_repair
from utils.inference_vllm import load_vllm_model, model_path_CoT, model_path_noCoT, schema_CoT, schema_noCoT
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.sampling_params import GuidedDecodingParams
import datetime

cached_path = "./cached_subject"
subject_path = "./jsons"
PROMPT = "你是一个精通 ACG 文化的影评分析专家。你的任务是分析用户的 Bangumi 评论及其情感倾向，并将其转化为符合以下定义的 JSON 格式。\n\n 输出要求:\n1. 必须严格遵守 JSON 语法，输出原始、紧凑的 JSON，不要输出 markdown 代码块（```json）。\n2. JSON 必须包含以下字段:\n - \"glossary\": {\"术语名\": \"解释\"}。如果评论中包含 ACG 行话或术语（如 \"神回\"、\"暴死\"），请在此解释；否则为空对象。\n - \"step_by_step_analysis\": 字符串。请先分析评论中提到的黑话，再结合 \"虽然... 但是...\" 等逻辑，分析评论中表达出对于剧情、作画、音乐、角色等维度的优缺点评价。\n - \"aspects\": 对象。包含 \"story\", \"visual\", \"music\",\"character\" 四个键，值为 \"Positive\", \"Negative\", \"Neutral\" 或 \"Not Mentioned\"。\n - \"final_sentiment\": 字符串。整体评价，仅限 \"Positive\", \"Neutral\", \"Negative\"。\n - \"confidence\": 整数。1-10，表示你对该判断的置信度。\n\n 注意：先进行分析（step_by_step_analysis），再给出结论。不要编造用户未提到的观点。\n\n"
PROMPT_CoT =  "你是一个精通 ACG 文化的影评分析专家。你的任务是分析用户的 Bangumi 评论及其情感倾向，并将其转化为符合以下定义的 JSON 格式。\n\n 输出要求:\n1. 必须严格遵守 JSON 语法，输出原始、紧凑的 JSON，不要包含任何空格或缩进，不要输出 markdown 代码块（```json），输出以{开头，字符串以英文双引号开头，且务必输出对应的双引号结尾。\n2. JSON 必须包含以下字段:\n - \"glossary\": {\"术语名\": \"解释\"}。如果评论中包含 ACG 行话或术语（如 \"神回\"、\"暴死\"），请在此解释；否则为空对象。\n - \"step_by_step_analysis\": 字符串。请先分析评论中提到的黑话，再结合 \"虽然... 但是...\" 等逻辑，分析评论中表达出对于剧情、作画、音乐、角色等维度的优缺点评价。\n - \"aspects\": 对象。包含 \"story\", \"visual\", \"music\",\"character\" 四个键，值为 \"Positive\", \"Negative\", \"Neutral\" 或 \"Not Mentioned\"。\n - \"final_sentiment\": 字符串。整体评价，仅限 \"Positive\", \"Neutral\", \"Negative\"。\n - \"confidence\": 整数。1-10，表示你对该判断的置信度。\n\n 注意：先进行分析（step_by_step_analysis），再给出结论。不要编造用户未提到的观点。\n\n"
aspects_ch = {"story": "剧情", "visual": "画面", "music": "音乐", "character": "角色", "final_sentiment": "总评价"}

def display_analysis(data):
    """展示单条分析结果（不包含外层标题，适用于当前结果和历史记录）"""
    st.markdown("**维度评分：**")
    cols = st.columns(5)
    aspect_map = data['aspects'].copy()
    aspect_map["final_sentiment"] = data.get("final_sentiment", "Not Mentioned")
    color_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪", "Not Mentioned": "⚫"}
    aspect_keys = ["visual", "story", "character", "music"]

    for idx, aspect_key in enumerate(aspect_keys):
        val = aspect_map.get(aspect_key, "Not Mentioned").capitalize()
        icon = color_map.get(val, "⚫")
        with cols[idx]:
            st.markdown(f"{aspects_ch[aspect_key]}")
            st.caption(f"{icon} {val}")
    # 最后一个列显示总评价
    val = aspect_map.get("final_sentiment", "Not Mentioned").capitalize()
    icon = color_map.get(val, "⚫")
    with cols[-1]:
        st.markdown(f"{aspects_ch['final_sentiment']}")
        st.caption(f"{icon} {val}")

    # 如果是 CoT 模式，额外显示术语和思维链
    if 'glossary' in data:
        if data.get('glossary'):
            st.markdown("**📖 术语识别：**")
            for term, desc in data['glossary'].items():
                st.success(f"**{term}**: {desc}")
        st.markdown("**🧠 深度思维链分析：**")
        st.markdown(f"> {data['analysis']}")

    st.caption(f"🤖 模型置信度: {data['confidence']}/10")

def analyze_comment(comment, model, with_CoT=False, api_key=None):
    prompt = PROMPT_CoT if with_CoT == "分析模式" else PROMPT
    if model == "在线":
        with st.spinner("在线调用模型分析中..."):
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": comment}
                    ],
                    response_format={'type': 'json_object'},
                    stream=False
                )
                # 解析结果并更新output
                res_content = response.choices[0].message.content
            except Exception as e:
                st.error(f"调用在线模型失败: {e}")
                return None
            try:
                res_json = json.loads(res_content)
            except json.JSONDecodeError:
                st.error("解析模型输出失败，可能是模型没有正确遵守输出格式要求，请重试。")
                return None
    else:
        model_path = model_path_CoT if with_CoT == "分析模式" else model_path_noCoT
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = load_vllm_model(model_path)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": comment}
        ]
        # 生成模型输入的 prompt 字符串
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 根据是否启用 CoT 选择 JSON schema
        schema = schema_CoT if with_CoT == "分析模式" else schema_noCoT
        
        # 设置 guided decoding 参数
        guided_decoding_params = GuidedDecodingParams(json=schema)
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            guided_decoding=guided_decoding_params
        )
        
        # 执行推理（使用 spinner 提供用户反馈）
        with st.spinner("本地模型推理中..."):
            try:
                outputs = llm.generate([prompt_text], sampling_params)
                generated_text = outputs[0].outputs[0].text
            except Exception as e:
                st.error(f"本地模型推理失败: {e}")
                return None
        
        # 解析 JSON 输出
        try:
            res_json = json.loads(generated_text)
        except json.JSONDecodeError:
            st.error("模型输出不是有效的 JSON，可能未遵守输出格式要求。")
            return None
    
    if with_CoT == "分析模式":
        sample_data = {
            "raw_text": comment,
            "analysis": res_json.get("step_by_step_analysis", "暂无分析"),
            "glossary": res_json.get("glossary", {}),
            "aspects": res_json.get("aspects", {}),
            "confidence": res_json.get("confidence", 0),
            "final_sentiment": res_json.get("final_sentiment", "Unknown")
        }
    else:
        sample_data = {
            "raw_text": comment,
            "aspects": res_json.get("aspects", {}),
            "confidence": res_json.get("confidence", 0),
            "final_sentiment": res_json.get("final_sentiment", "Unknown")
        }
    st.subheader("分析结果")
    display_analysis(sample_data)
    return sample_data



if __name__ == "__main__":
    st.set_page_config(page_title="Bangumi 评论分析系统", layout="wide")
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.title("单条评论分析")
    st.markdown("输入评论，综合分析评论的情感倾向。")

    with st.form("input_form"):
        # 第一行：评论输入框（多行文本）
        comment = st.text_area(
            "评论内容",
            placeholder="请输入要分析的评论...",
            height=150,
            label_visibility="collapsed"
        )

        # 第二行：模型选择、分析模式、API Key 并排
        col1, col2, col3 = st.columns([2, 2, 3])  # 调整列宽比例，API Key 稍宽
        model = col1.selectbox(
            "模型选择",
            ["本地", "在线"],
            label_visibility="collapsed"
        )
        with_CoT = col2.pills(
            "是否启用CoT",
            ["分析模式"],
            selection_mode="single",
            label_visibility="collapsed"
        )
        api_key = col3.text_input(
            "API Key",
            placeholder="选择在线模式时请输入 API Key",
            label_visibility="collapsed"
        )

        # 提交按钮（宽度自适应）
        submitted = st.form_submit_button("🔍 分析", use_container_width=True)
        
           
        if submitted:
            if not comment.strip():
                st.warning("请输入评论内容后再提交分析。")
            elif model == "在线" and not api_key.strip():
                st.warning("请选择在线模式时请输入 API Key。")
            else:
                result = analyze_comment(comment, model=model, with_CoT=with_CoT, api_key=api_key)
                if result is not None:   # 分析成功
                    result['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append(result)
        st.markdown("---")
        st.subheader("📜 历史分析记录")

        if not st.session_state.history:
            st.info("暂无历史记录")
        else:
            # 倒序显示（最新的在上）
            for idx, item in enumerate(reversed(st.session_state.history)):
                # 截断评论作为标题
                short_comment = item['raw_text'][:50] + "..." if len(item['raw_text']) > 50 else item['raw_text']
                with st.expander(f"{item['timestamp']} - {short_comment}"):
                    # 在 expander 内部直接调用 display_analysis，不再加额外标题
                    display_analysis(item)

