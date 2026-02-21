import streamlit as st



st.sidebar.markdown("### BGM Bot - 轻松分析你的评论！")
pg = st.navigation([st.Page("subject.py", title="条目批量分析"), st.Page("single_comment.py", title="单条评论分析")])
pg.run()