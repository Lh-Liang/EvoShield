"""
LLM Client for EvoShield
使用 OpenAI API 调用 LLM 进行预测
"""
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from openai import OpenAI, OpenAIError

# 尝试加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # 如果没有安装 python-dotenv，使用环境变量

# 从 prompt.py 导入提示模板和配置
from prompt import build_prompt, TASK_CATEGORIES, LLM_TEMPERATURE


class LLMClient:
    def __init__(self, model, api_key: str = None, base_url: str = None):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: OpenAI API key，默认从环境变量或 .env 文件获取
            base_url: API 基础 URL，默认从环境变量或 .env 文件获取
            model: 使用的模型名称，默认从环境变量或 .env 文件获取
        """
        # 优先使用传入的参数，否则从环境变量读取
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY', 'sk-xxxx')
        if base_url is None:
            base_url = os.environ.get('OPENAI_BASE_URL', 'https://api2.aigcbest.top/v1')
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def predict(self, text: str, task_name: str, categories: dict) -> Optional[int]:
        """
        使用 LLM 预测文本类别
        
        Args:
            text: 输入文本
            task_name: 任务名称 (PI, JC, SG)
            categories: 类别映射字典，默认根据 task_name 从 prompt.py 获取
            
        Returns:
            预测的类别索引；若无法解析则返回 None
        """
        # 如果没有提供类别，则根据任务名称从 prompt.py 获取
        if categories is None:
            categories = TASK_CATEGORIES.get(task_name, TASK_CATEGORIES["PI"])
        
        # 使用 prompt.py 中的 build_prompt 函数构建提示
        prompt = build_prompt(text, categories)

        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=LLM_TEMPERATURE,
                )
                break
            except OpenAIError as exc:
                if attempt == self.max_retries:
                    logger.warning(
                        f"LLM 请求连续失败 {self.max_retries} 次，已跳过当前样本: "
                        f"model={self.model}, task={task_name}, error={exc}"
                    )
                    return None
                logger.warning(
                    f"LLM 请求失败，第 {attempt}/{self.max_retries} 次重试: "
                    f"model={self.model}, task={task_name}, error={exc}"
                )
                time.sleep(self.retry_delay)

        result = response.choices[0].message.content.strip()

        predicted_label = self._parse_label(result, categories)
        if predicted_label is not None:
            return predicted_label

        print(f"Warning: LLM 返回了无法解析的结果 '{result}'，将跳过该样本")
        return None

    def _parse_label(self, result: str, categories: dict) -> Optional[int]:
        max_label = len(categories) - 1

        try:
            predicted_label = int(result)
            if 0 <= predicted_label <= max_label:
                return predicted_label
        except ValueError:
            pass

        if result:
            last_char = result[-1]
            if last_char.isdigit():
                predicted_label = int(last_char)
                if 0 <= predicted_label <= max_label:
                    return predicted_label

        return None
    
    def batch_predict(self, texts: list, task_name: str = "PI", categories: dict = None) -> list:
        """
        批量预测文本类别
        
        Args:
            texts: 输入文本列表
            task_name: 任务名称
            categories: 类别映射字典
            
        Returns:
            预测的类别索引列表
        """
        return [self.predict(text, task_name, categories) for text in texts]


def get_llm_client(model) -> LLMClient:
    """
    获取 LLM 客户端实例的便捷函数
    
    Args:
        api_key: API key
        
    Returns:
        LLMClient 实例
    """
    return LLMClient(model=model)


if __name__ == "__main__":
    # 测试代码
    client = LLMClient(model='grok-4.1')

    PI_CATEGORIES = {
        0: "benign",
        1: "injection"
    }

    test_text = "The company announced record profits for the third quarter."
    predicted = client.predict(test_text, task_name="PI", categories=PI_CATEGORIES)
    print(f"Predicted label: {predicted}")
