import datetime
import streamlit as st
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import SamplingParams

from single_comment_config import ASPECTS_CH, DEFAULT_FALLBACK_STRATEGY, PROMPT
from single_comment_rag import (
    build_comment_context,
    normalize_source_title,
    parse_json_response,
    run_agentic_rag_analysis,
)
from utils.inference_vllm import load_vllm_model, model_path_CoT, model_path_noCoT


def display_analysis(data):
    if data.get("source_title"):
        st.caption(f"🎬 来源作品: {data['source_title']}")
    st.markdown("**维度评分：**")
    cols = st.columns(5)
    aspect_map = data["aspects"].copy()
    aspect_map["final_sentiment"] = data.get("final_sentiment", "Not Mentioned")
    color_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪", "Not Mentioned": "⚫"}
    aspect_keys = ["visual", "story", "character", "music"]

    for idx, aspect_key in enumerate(aspect_keys):
        val = aspect_map.get(aspect_key, "Not Mentioned").capitalize()
        icon = color_map.get(val, "⚫")
        with cols[idx]:
            st.markdown(f"{ASPECTS_CH[aspect_key]}")
            st.caption(f"{icon} {val}")

    val = aspect_map.get("final_sentiment", "Not Mentioned").capitalize()
    icon = color_map.get(val, "⚫")
    with cols[-1]:
        st.markdown(f"{ASPECTS_CH['final_sentiment']}")
        st.caption(f"{icon} {val}")

    if "glossary" in data:
        if data.get("glossary"):
            st.markdown("**📖 术语识别：**")
            for term, desc in data["glossary"].items():
                st.success(f"**{term}**: {desc}")
        st.markdown("**🧠 深度思维链分析：**")
        st.markdown(f"> {data.get('analysis', '暂无分析')}")

    st.caption(f"🤖 模型置信度: {data['confidence']}/10")


class BaseLLM:
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        force_json: bool = False,
    ) -> str:
        raise NotImplementedError

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: dict,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict:
        raw = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            force_json=True,
        )
        try:
            return parse_json_response(raw)
        except Exception:
            return fallback


class OnlineLLM(BaseLLM):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        force_json: bool = False,
    ) -> str:
        kwargs = {"stream": False, "reasoning_effort": "high"}
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content or ""


@st.cache_resource
def load_tokenizer(path: str):
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


class VllmLLM(BaseLLM):
    def __init__(self, model_path: str):
        self.tokenizer = load_tokenizer(model_path)
        self.llm = load_vllm_model(model_path)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        force_json: bool = False,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate([prompt_text], sampling_params)
        return outputs[0].outputs[0].text.strip()


def analyze_plain_comment(llm: BaseLLM, comment: str, source_title: str = "") -> dict:
    fallback = {
        "glossary": {},
        "step_by_step_analysis": "模型输出解析失败，未能生成详细分析。",
        "aspects": {
            "story": "Not Mentioned",
            "visual": "Not Mentioned",
            "music": "Not Mentioned",
            "character": "Not Mentioned",
        },
        "final_sentiment": "Neutral",
        "confidence": 5,
    }
    user_prompt = build_comment_context(comment, source_title)
    return llm.chat_json(PROMPT, user_prompt, fallback=fallback, max_new_tokens=900, temperature=0.2)


def analyze_comment(
    comment: str,
    model: str,
    with_CoT=False,
    api_key: str = "",
    fallback_strategy: str = DEFAULT_FALLBACK_STRATEGY,
    tavily_api_key: str = "",
    source_title: str = "",
):
    is_rag_mode = with_CoT == "分析模式"
    normalized_source_title = normalize_source_title(source_title)

    try:
        if model == "在线":
            llm: BaseLLM = OnlineLLM(api_key=api_key)
            spinner_text = "在线模型分析中..."
        else:
            model_path = model_path_CoT if is_rag_mode else model_path_noCoT
            llm = VllmLLM(model_path)
            spinner_text = "本地 vLLM 模型分析中..."

        with st.spinner(spinner_text):
            if is_rag_mode:
                res_json = run_agentic_rag_analysis(
                    llm=llm,
                    comment=comment,
                    source_title=normalized_source_title,
                    fallback_strategy=fallback_strategy,
                    tavily_api_key=tavily_api_key,
                )
            else:
                res_json = analyze_plain_comment(llm, comment, normalized_source_title)
    except Exception as exc:
        st.error(f"分析失败: {exc}")
        return None

    if is_rag_mode:
        sample_data = {
            "raw_text": comment,
            "source_title": normalized_source_title,
            "analysis": res_json.get("step_by_step_analysis", "暂无分析"),
            "glossary": res_json.get("glossary", {}),
            "aspects": res_json.get("aspects", {}),
            "confidence": res_json.get("confidence", 0),
            "final_sentiment": res_json.get("final_sentiment", "Unknown"),
        }
    else:
        sample_data = {
            "raw_text": comment,
            "source_title": normalized_source_title,
            "aspects": res_json.get("aspects", {}),
            "confidence": res_json.get("confidence", 0),
            "final_sentiment": res_json.get("final_sentiment", "Unknown"),
        }

    st.subheader("分析结果")
    display_analysis(sample_data)
    return sample_data


if __name__ == "__main__":
    st.set_page_config(page_title="Bangumi 评论分析系统", layout="wide")
    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("单条评论分析")
    st.markdown("输入评论，综合分析评论的情感倾向。勾选“分析模式”时会启用 RAG。")

    with st.form("input_form"):
        comment = st.text_area(
            "评论内容",
            placeholder="请输入要分析的评论...",
            height=150,
            label_visibility="collapsed",
        )

        col1, col2, col3 = st.columns([2, 2, 3])
        model = col1.selectbox("模型选择", ["本地", "在线"], label_visibility="collapsed")
        with_CoT = col2.pills(
            "是否启用CoT",
            ["分析模式"],
            selection_mode="single",
            label_visibility="collapsed",
        )
        fallback_strategy = col3.selectbox(
            "Fallback Strategy",
            ["先进行全文搜索再进行网页搜索", "先进行网页搜索再进行全文搜索", "只进行全文搜索", "只进行网页搜索"],
            index=0,
            label_visibility="collapsed",
        )

        col4, col5, col6 = st.columns([3, 3, 3])
        source_title = col4.text_input(
            "来源作品名",
            placeholder="可选：输入评论对应的番剧/作品名",
            label_visibility="collapsed",
        )
        api_key = col5.text_input(
            "API Key",
            placeholder="选择在线模式时请输入 DeepSeek API Key",
            label_visibility="collapsed",
        )
        tavily_api_key = col6.text_input(
            "Tavily API Key",
            placeholder="Tavily Key（为空或无效时自动回退到 ddgs）",
            label_visibility="collapsed",
            type="password",
        )

        submitted = st.form_submit_button("🔍 分析", use_container_width=True)

        if submitted:
            if not comment.strip():
                st.warning("请输入评论内容后再提交分析。")
            elif model == "在线" and not api_key.strip():
                st.warning("在线模式需要输入 API Key。")
            else:
                result = analyze_comment(
                    comment,
                    model=model,
                    with_CoT=with_CoT,
                    api_key=api_key,
                    fallback_strategy=fallback_strategy,
                    tavily_api_key=tavily_api_key,
                    source_title=source_title,
                )
                if result is not None:
                    result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append(result)

        st.caption("说明：只有勾选“分析模式”时才会启用 RAG；Tavily Key 为空或不可用时，网页补召回会自动回退到 ddgs。")
        st.markdown("---")
        st.subheader("📜 历史分析记录")

        if not st.session_state.history:
            st.info("暂无历史记录")
        else:
            for item in reversed(st.session_state.history):
                short_comment = item["raw_text"][:50] + "..." if len(item["raw_text"]) > 50 else item["raw_text"]
                with st.expander(f"{item['timestamp']} - {short_comment}"):
                    display_analysis(item)
