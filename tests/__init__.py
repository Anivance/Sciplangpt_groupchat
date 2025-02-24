"""Test configuration and utilities."""

import os
import sys
from pathlib import Path

# 添加專案根目錄到 Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 測試環境變量設置
os.environ.setdefault('TESTING', 'True')
os.environ.setdefault('POSTGRES_DB', 'agent_db_test')
os.environ.setdefault('REDIS_DB', '1')

# 測試用的常量
TEST_DATA_DIR = ROOT_DIR / 'tests' / 'test_data'
TEST_FILES_DIR = ROOT_DIR / 'tests' / 'test_files'

# 確保測試目錄存在
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)