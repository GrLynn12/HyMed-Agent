"""用户凭证的 JSON 存储。

.. warning::
   **SECURITY**: 当前实现把密码以**明文**形式写入 ``tmp_data/user_credentials.json``，
   仅适用于本地 demo / 实验环境。生产部署前必须替换为带加盐哈希的方案
   （推荐 ``passlib`` 的 ``bcrypt`` / ``argon2``）。
"""

from __future__ import annotations

import json
import os
from typing import Dict


class Credentials:
    """单个用户的凭证记录。"""

    def __init__(self, username: str, password: str, is_admin: bool = False) -> None:
        self.username = username
        self.password = password  # SECURITY: 明文存储，仅供 demo
        self.is_admin = is_admin

    def to_dict(self) -> Dict[str, object]:
        return {
            'username': self.username,
            'password': self.password,
            'is_admin': self.is_admin,
        }


def create_folder_if_not_exist(folder: str) -> None:
    """目录不存在则创建。"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def read_credentials(file_path: str) -> Dict[str, Credentials]:
    """读取并反序列化全部用户凭证；文件不存在或损坏时返回空字典。"""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            return {k: Credentials(**v) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_credentials(file_path: str, credentials_dict: Dict[str, Credentials]) -> None:
    """把用户凭证字典写回 JSON 文件。"""
    data = {k: v.to_dict() for k, v in credentials_dict.items()}
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


# 文件存储位置
storage_folder = "tmp_data"
storage_file = os.path.join(storage_folder, "user_credentials.json")

# 确保文件夹存在
create_folder_if_not_exist(storage_folder)

# 读取现有的用户数据
credentials: Dict[str, Credentials] = read_credentials(storage_file)

# 如果初始文件为空，则初始化管理员账户
if not credentials:
    admin = Credentials("admin", "admin123", True)
    credentials['admin'] = admin
    write_credentials(storage_file, credentials)
