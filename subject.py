import streamlit as st
import os
from utils.llm_api_aysnc import process_single_file
import json
import plotly.graph_objects as go
import json_repair
import random
import asyncio
from utils.inference_vllm import inference_with_vllm

cached_path = "./cached_subject"
subject_path = "./jsons"
PROMPT = "你是一个精通 ACG 文化的影评分析专家。你的任务是分析用户的 Bangumi 评论及其情感倾向，并将其转化为符合以下定义的 JSON 格式。\n\n 输出要求:\n1. 必须严格遵守 JSON 语法，输出原始、紧凑的 JSON，不要输出 markdown 代码块（```json）。\n2. JSON 必须包含以下字段:\n - \"glossary\": {\"术语名\": \"解释\"}。如果评论中包含 ACG 行话或术语（如 \"神回\"、\"暴死\"），请在此解释；否则为空对象。\n - \"step_by_step_analysis\": 字符串。请先分析评论中提到的黑话，再结合 \"虽然... 但是...\" 等逻辑，分析评论中表达出对于剧情、作画、音乐、角色等维度的优缺点评价。\n - \"aspects\": 对象。包含 \"story\", \"visual\", \"music\",\"character\" 四个键，值为 \"Positive\", \"Negative\", \"Neutral\" 或 \"Not Mentioned\"。\n - \"final_sentiment\": 字符串。整体评价，仅限 \"Positive\", \"Neutral\", \"Negative\"。\n - \"confidence\": 整数。1-10，表示你对该判断的置信度。\n\n 注意：先进行分析（step_by_step_analysis），再给出结论。不要编造用户未提到的观点。\n\n"
PROMPT_CoT =  "你是一个精通 ACG 文化的影评分析专家。你的任务是分析用户的 Bangumi 评论及其情感倾向，并将其转化为符合以下定义的 JSON 格式。\n\n 输出要求:\n1. 必须严格遵守 JSON 语法，输出原始、紧凑的 JSON，不要包含任何空格或缩进，不要输出 markdown 代码块（```json），输出以{开头，字符串以英文双引号开头，且务必输出对应的双引号结尾。\n2. JSON 必须包含以下字段:\n - \"glossary\": {\"术语名\": \"解释\"}。如果评论中包含 ACG 行话或术语（如 \"神回\"、\"暴死\"），请在此解释；否则为空对象。\n - \"step_by_step_analysis\": 字符串。请先分析评论中提到的黑话，再结合 \"虽然... 但是...\" 等逻辑，分析评论中表达出对于剧情、作画、音乐、角色等维度的优缺点评价。\n - \"aspects\": 对象。包含 \"story\", \"visual\", \"music\",\"character\" 四个键，值为 \"Positive\", \"Negative\", \"Neutral\" 或 \"Not Mentioned\"。\n - \"final_sentiment\": 字符串。整体评价，仅限 \"Positive\", \"Neutral\", \"Negative\"。\n - \"confidence\": 整数。1-10，表示你对该判断的置信度。\n\n 注意：先进行分析（step_by_step_analysis），再给出结论。不要编造用户未提到的观点。\n\n"
aspects_ch = {"story": "剧情", "visual": "画面", "music": "音乐", "character": "角色", "final_sentiment": "总评价"}

def display_analysis_report(metrics_data, overall_avg):
    """
    metrics_data 格式: 
    { 'Visual': {'avg_score': 8.5, 'mention_rate': 0.75}, ... }
    """
    st.title("📊 评论维度深度分析")
    
    # --- 第一部分：顶部核心指标 (Final Sentiment Score) ---
    st.markdown("### 综合评价得分")
    st.metric(label="情感综合分", value=f"{overall_avg:.2f} / 10")
    st.divider()

    # --- 第二部分：左图右表布局 ---
    col_chart, col_metrics = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown("#### 分数分布雷达图")
        
        categories = list(metrics_data.keys())
        # 提取平均分进行绘图
        scores = [d['avg_score'] for d in metrics_data.values()]
        
        # 闭合雷达图
        plot_categories = categories + [categories[0]]
        plot_scores = scores + [scores[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=plot_scores,
            theta=plot_categories,
            fill='toself',
            fillcolor='rgba(255, 75, 75, 0.2)',  # 浅红色填充
            mode='lines+markers',
            marker=dict(size=8, color='#FF4B4B'), # 红色圆点
            line=dict(color='#FF4B4B', width=3),  # 红色线条
            name='维度平均分'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    gridcolor="lightgrey",
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    gridcolor="lightgrey",
                    tickfont=dict(size=12, color="gray")
                )
            ),
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40),
            height=450,
            paper_bgcolor='rgba(0,0,0,0)', # 背景透明
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.markdown("#### 详细数据统计")
        for aspect, data in metrics_data.items():
            # 使用 Streamlit 的容器美化每一行数据
            with st.container():
                # 创建子列展示指标
                
                st.markdown(f"### {aspect}")
                m1, m2 = st.columns(2)
                m1.metric("平均分", f"{data['avg_score']:.1f}")
                m2.metric("提及率", f"{data['mention_rate']:.1%}")
                st.write("---") # 分割线

def analyze_subject(cached_file_path, with_CoT, results=None):
    if results is None:
        with open(cached_file_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
    aspect_keys = ["visual", "story", "character", "music"]
    score_map = {"positive": 10, "neutral": 6, "negative": 2}
    samples_by_sentiment = {
        "positive": [],
        "negative": [],
        "neutral": []
    }
    
    # 初始化统计
    stats = {key: {"total_score": 0, "count": 0} for key in aspect_keys}
    final_sentiment_stats = {"total_score": 0, "count": 0}
    total_items = len(results)

    for item in results:
        try:
            data = json_repair.loads(item.get("output", "{}"))
            
            # 1. 计算总情感得分 (final_sentiment)
            f_sentiment = data.get("final_sentiment").lower()
            if f_sentiment in score_map:
                final_sentiment_stats["total_score"] += score_map[f_sentiment]
                final_sentiment_stats["count"] += 1
            
            # 2. 计算各维度得分
            aspects = data.get("aspects", {})
            for key in aspect_keys:
                sentiment = aspects.get(key, "").lower()
                if sentiment in score_map:
                    stats[key]["count"] += 1
                    stats[key]["total_score"] += score_map[sentiment]

            if f_sentiment in samples_by_sentiment:
                if with_CoT == "分析模式":
                    sample_data = {
                        "raw_text": item.get("input", ""),
                        "analysis": data.get("step_by_step_analysis", "暂无分析"),
                        "glossary": data.get("glossary", {}),
                        "aspects": aspects,
                        "confidence": data.get("confidence", 0),
                        "final_sentiment": f_sentiment.capitalize()
                    }
                    samples_by_sentiment[f_sentiment].append(sample_data)
                else:
                    sample_data = {
                        "raw_text": item.get("input", ""),
                        "aspects": aspects,
                        "confidence": data.get("confidence", 0),
                        "final_sentiment": f_sentiment.capitalize()
                    }
                    samples_by_sentiment[f_sentiment].append(sample_data)
        except:
            continue

    # 计算最终指标
    categories = ["画面", "剧情", "角色", "音乐"]
    metrics = {}
    
    # 计算总平均分
    if final_sentiment_stats["count"] > 0:
        overall_avg = final_sentiment_stats["total_score"] / final_sentiment_stats["count"]
        # print(f"全站综合得分 (Final Sentiment): {overall_avg:.2f}")

    for key in aspect_keys:
        count = stats[key]["count"]
        avg = stats[key]["total_score"] / count if count > 0 else 0
        mention_rate = count / total_items
        
        # categories.append(key.capitalize())
        metrics[aspects_ch.get(key, key)] = {"avg_score": avg, "mention_rate": mention_rate}
    categories.append("总情感倾向")
    display_analysis_report(metrics, overall_avg)

    st.markdown("---")
    st.subheader("🔍 模型深度解析实录")
    st.caption("从数千条评论中随机抽取典型样本，展示模型对剧情、黑话及情感的理解能力。")

    # 使用 Tabs 分类展示
    tab1, tab2, tab3 = st.tabs(["👍 好评解析", "👎 差评解析", "⚖️ 中立/复杂解析"])

    def render_samples(sentiment_key, color_theme):
        """渲染评论列表的辅助函数"""
        samples = samples_by_sentiment.get(sentiment_key, [])
        if not samples:
            st.info("该类别暂无评论样本。")
            return
        
        # 随机抽取 3 条，避免刷屏
        display_samples = random.sample(samples, min(len(samples), 3))
        
        for i, s in enumerate(display_samples):
            # 外层使用 Expander，标题显示评论摘要
            with st.expander(f"💬 评论 {i+1}: {s['raw_text'][:30]}...", expanded=(i==0)):
                
                # 第一行：原始评论引用
                st.markdown(f"**用户原评：**")
                st.info(s['raw_text'])
                
                # 第二行：细粒度评分展示 (用 Columns 布局)
                st.markdown("**维度评分：**")
                cols = st.columns(5)
                aspect_map = s['aspects']
                aspect_map["final_sentiment"] = s.get("final_sentiment", "Not Mentioned")
                # 定义颜色映射
                color_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪", "Not Mentioned": "⚫"}
                
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

                if with_CoT == "分析模式":
                    # 第三行：黑话解释 (如果有)
                    if s['glossary']:
                        st.markdown("**📖 术语识别：**")
                        for term, desc in s['glossary'].items():
                            st.success(f"**{term}**: {desc}")
                    
                    # 第四行：深度分析 (CoT)
                    st.markdown("**🧠 深度思维链分析：**")
                    st.markdown(f"> {s['analysis']}")
                
                # 底部：置信度
                st.caption(f"🤖 模型置信度: {s['confidence']}/10")
    
    with tab1:
        render_samples("positive", "green")
    with tab2:
        render_samples("negative", "red")
    with tab3:
        render_samples("neutral", "grey")
        


def generate_response(file_path, cached_file_path, model, api_key, with_CoT):
    instruction = PROMPT_CoT if with_CoT == "分析模式" else PROMPT
    if model == "在线":
        st_bar = st.progress(0, text="准备调用 API...")
        
        # --- 关键修改：驱动异步函数 ---
        try:
            # 运行异步处理逻辑
            results = asyncio.run(process_single_file(
                file_path, cached_file_path, api_key, instruction, st_bar
            ))
            
            if results:
                st.success(f"处理完成，共 {len(results)} 条记录")
                analyze_subject(cached_file_path, with_CoT, results)
        except Exception as e:
            st.error(f"处理过程中出现异常: {e}")
    elif model == "本地":
        # 调用本地模型进行分析，生成 JSONL 文件
        st_bar = st.progress(0, text="准备调用 本地模型...")
        try:
            results = inference_with_vllm(
                file_path, cached_file_path, instruction, st_bar, with_CoT, batch_size=1000
            )
            if results:
                st.success(f"处理完成，共 {len(results)} 条记录")
                analyze_subject(cached_file_path, with_CoT, results)
        except Exception as e:
            st.error(f"处理过程中出现异常: {e}")
    # analyze_subject(cached_file_path, with_CoT, results)

if __name__ == "__main__":
    st.set_page_config(page_title="Bangumi 评论分析系统", layout="wide")
    

    st.title("条目评论分析")
    st.markdown("输入条目ID，综合分析条目评论的情感倾向。")


    with st.form("input_form"):
        subject_input, model_select, thinking_mode = st.columns([7, 2, 2])
        api_key = None


        subject_id = subject_input.text_input("ID", placeholder="如链接为https://bgm.tv/subject/123，则输入123", label_visibility="collapsed")
        model = model_select.selectbox("模型选择", ["本地", "在线"], label_visibility="collapsed")
        with_CoT = thinking_mode.pills("是否启用CoT", ["分析模式"], selection_mode="single", label_visibility="collapsed", width="stretch")
        api_key =  st.text_input("API Key", placeholder="选择在线模式时 请输入在线模型的 API Key", label_visibility="collapsed")
        submitted = st.form_submit_button("🔍")
        
           
        if submitted:
            if not subject_id.isdigit() and subject_id != "demo":
                st.warning("请输入有效的条目ID！")
            elif model == "在线" and not api_key:
                st.warning("请选择在线模式时请输入 API Key。")
            else:
                print(f"正在分析条目 {subject_id}，模型: {model}, CoT: {with_CoT}")
                cached_file_path = os.path.join(cached_path, f"{subject_id}_cot.jsonl") if with_CoT == "分析模式" else os.path.join(cached_path, f"{subject_id}.jsonl")
                if os.path.exists(cached_file_path):
                    st.success(f"使用缓存加载")
                    analyze_subject(cached_file_path, with_CoT)
                else:
                    file_path = os.path.join(subject_path, f"{subject_id}.jsonl")
                    if os.path.exists(file_path):
                        st.info(f"正在分析条目 {subject_id} 的评论...")
                        generate_response(file_path, cached_file_path, model, api_key, with_CoT)
                    else:
                        # crawler(subject_id)
                        st.error(f"未找到条目 {subject_id} 的评论数据，请确保已爬取该条目的评论并放置在 {subject_path} 目录下。")

