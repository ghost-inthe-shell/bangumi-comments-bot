import requests
import random
import re
import json
import os
import time
from datetime import datetime
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# 配置信息
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/5.0 (Macintosh; U; PPC Mac OS X 10.5; en-US; rv:1.9.2.15) Gecko/20110303 Firefox/3.6.15",
]
SUBJECT_IDS = [514358] # 把条目ID放在这里，支持批量爬取多个条目
OUTPUT_DIR = "../jsons"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def scrape_subject(subject_id):
    """处理单个条目的所有评论爬取"""
    url = f"https://bgm.tv/subject/{subject_id}/comments"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    
    try:
        # 获取总页数
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")
        
        page_edge = soup.find("span", class_="p_edge")
        if not page_edge: return # 无评论或页面错误
        
        match = re.search(r'/\s*(\d+)', page_edge.get_text())
        page_num = int(match.group(1)) if match else 1
        
        all_comments = []
        
        # 遍历每一页
        for page in range(1, page_num + 1):
            page_url = f"{url}?page={page}"
            page_res = requests.get(page_url, headers=headers, timeout=10)
            page_res.encoding = "utf-8"
            page_soup = BeautifulSoup(page_res.text, "html.parser")
            
            # 解析评论项
            items = page_soup.find_all("div", class_="item clearit")
            current_page_comments = []
            
            for item in items:
                small_tags = item.find_all("small", class_="grey")
                # 过滤只看“看过”的评论
                if not small_tags or "看过" not in small_tags[0].get_text():
                    continue
                
                comment = {
                    "timestamp": small_tags[1].get_text(strip=True) if len(small_tags) > 1 else None,
                    "stars": next((int(cls[5:]) for cls in item.find("span", class_="starlight").get("class", []) if cls.startswith("stars")), None) if item.find("span", class_="starlight") else None,
                    "text": item.find("p", class_="comment").get_text(strip=True) if item.find("p", class_="comment") else None,
                    "id": item.find("div", class_="likes_grid").get("id", "").replace("likes_grid_", "") if item.find("div", class_="likes_grid") else None
                }
                current_page_comments.append(comment)

            # 解析点赞数据 (JavaScript 提取)
            likes_match = re.search(r'var data_likes_list = ({.*?});', page_res.text, re.DOTALL)
            if likes_match:
                try:
                    likes_data = json.loads(likes_match.group(1))
                    for c in current_page_comments:
                        cid = str(c['id'])
                        if cid in likes_data:
                            raw_likes = likes_data[cid]
                            # 处理 likes 可能为 list 或 dict 的情况
                            iterable = raw_likes.values() if isinstance(raw_likes, dict) else raw_likes
                            c["likes"] = {l.get("emoji"): l.get("total") for l in iterable}
                except: pass

            all_comments.extend(current_page_comments)
            if page % 10 == 0:
                print(f"[{subject_id}] 已爬取到第 {page} 页")
            time.sleep(1.0) # 单线程内部微小延迟，保护服务器
            
        # 保存文件
        with open(f"{OUTPUT_DIR}/{subject_id}.jsonl", "w", encoding="utf-8") as f:
            for item in all_comments:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
        print(f"完成 [{subject_id}]: 共 {len(all_comments)} 条评论")
        
    except Exception as e:
        print(f"处理 [{subject_id}] 出错: {e}")

# --- 执行入口 ---
if __name__ == "__main__":
    print(f"开始爬取，任务总数: {len(SUBJECT_IDS)}")
    
    # max_workers 建议不要设置太大（3-5即可），以免被 Bangumi 封禁 IP
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(scrape_subject, SUBJECT_IDS)
    
    print("所有任务已完成")