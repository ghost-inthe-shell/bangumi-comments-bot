import json
import asyncio
from collections import deque
from tqdm import tqdm
from openai import AsyncOpenAI  # 替换为异步客户端
import streamlit as st

# ===================== 配置项 =====================
BASE_URL = "https://api.deepseek.com"
SEMAPHORE_NUM = 2  # 并发数（根据API限流调整）
BUFFER_SIZE = 50    # 缓冲区大小：达到该值就写入文件
MAX_RETRIES = 2     # API调用最大重试次数
# ==================================================


# 通用写入文件函数（同步，因为文件写入本身是IO密集且异步收益低）
def write_to_file(data_list, file_path):
    """将数据写入文件（追加式，保持原有结构）"""
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # logger.info(f"缓冲区数据已写入文件: {file_path} (本次写入{len(data_list)}条)")

async def get_response_with_retry(entry, semaphore, client, instruction):
    """
    异步调用DeepSeek API，带指数退避重试
    :param entry: 单条数据条目（包含instruction/input/output）
    :param semaphore: 并发控制信号量
    :return: 处理后的entry（更新output）或原entry（失败时）
    """
    # instruction = entry["instruction"]
    input_content = entry["text"]
    
    for attempt in range(MAX_RETRIES):
        async with semaphore:  # 并发控制
            try:
                # 2. 异步调用DeepSeek API（核心修改：加await）
                response = await client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": input_content}
                    ],
                    response_format={'type': 'json_object'},
                    stream=False
                )
                # 解析结果并更新output
                res_content = response.choices[0].message.content
                res_dict = {
                    "instruction": instruction,
                    "input": input_content,
                    "output": res_content
                }
                # logger.debug(f"成功处理条目: {input_content[:15]}...")
                return res_dict  # 返回处理结果（字典形式）
            
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    # logger.error(f"重试{MAX_RETRIES}次后仍失败 | 条目: {input_content[:15]}... | 错误: {e}")
                    res_dict = {
                        "instruction": instruction,
                        "input": input_content,
                        "output": "{}",
                    }
                    return res_dict  # 返回错误信息（字典形式）
                # 指数退避重试（异步sleep，不阻塞其他任务）
                retry_delay = 2 ** attempt
                # logger.warning(f"第{attempt+1}次调用失败 | 延迟{retry_delay}s重试 | 错误: {e}")
                await asyncio.sleep(retry_delay)

async def process_single_file(file_path, cached_file_path, api_key, instruction, st_bar):
    """异步处理单个JSON文件"""
    # logger.info(f"开始处理文件: {file_path}")
    all_data = []
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )
    
    # 1. 读取原始数据
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = []
        for line in f:
            raw_data.append(json.loads(line))
    
    total_pending = len(raw_data)
    
    if total_pending == 0:
        # logger.info(f"文件{file_path}无待处理条目，跳过")
        return None
    
    # 初始化并发控制、锁、缓冲区、进度条
    semaphore = asyncio.Semaphore(SEMAPHORE_NUM)
    buffer_lock = asyncio.Lock()  # 保证缓冲区修改线程安全
    buffer = deque()  # 临时缓冲区：存储已处理的条目
    pbar = tqdm(total=total_pending, desc=f"处理{file_path}")
    
    # 2. 创建异步任务列表
    tasks = [get_response_with_retry(entry, semaphore, client, instruction) for entry in raw_data]
    
    # 3. 按完成顺序处理任务
    for future in asyncio.as_completed(tasks):
        result_dict = await future
        async with buffer_lock:
            # 将处理后的条目加入缓冲区
            if result_dict is not None:
                buffer.append(result_dict)
            pbar.update(1)
            if pbar.n % 10 == 0 or pbar.n == total_pending:
                st_bar.progress(pbar.n / total_pending, text=f"在线调用api处理中，进度: {pbar.n}/{total_pending}")  # 更新streamlit进度条
            
            # 缓冲区达到阈值时，写入文件（合并已处理+缓冲区+剩余未处理）
            if len(buffer) >= BUFFER_SIZE:
                # 写入文件（追加式，保持原有结构）
                write_to_file(buffer, cached_file_path)
                all_data.extend(buffer)  # 将缓冲区数据加入总结果列表
                # 清空缓冲区（已写入）
                buffer.clear()
    
    # 4. 处理剩余缓冲区数据（任务全部完成后）
    async with buffer_lock:
        if buffer:
            write_to_file(buffer, cached_file_path)
            all_data.extend(buffer)
            buffer.clear()
    
    pbar.close()

    if len(all_data) != total_pending:
        # logger.warning(f"处理完成，但成功条目数({len(all_data)})与总条目数({total_pending})不匹配！")
        pass

    return all_data
    # logger.info(f"文件处理完成: {file_path}")
