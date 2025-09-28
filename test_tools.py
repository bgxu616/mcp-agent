
from search_tools import multi_search
from reasoning_tools import deep_reason

from dotenv import load_dotenv
import os
from dotenv import load_dotenv
import os

# 显式指定 .env 文件路径（比默认路径更可靠）
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(load_dotenv(dotenv_path=dotenv_path))
# 加载 .env 文件
# print(load_dotenv())
# 搜索
context=""
search_result = multi_search("NVIDIA股票分析", context,n_wide=3)

# 推理
reason_result = deep_reason("NVIDIA分析", search_result, n_deep=3)