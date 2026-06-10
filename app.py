"""HyMed-Agent 的 Streamlit 应用入口。

启动方式::

    streamlit run app.py

登录成功后会进入医疗问答工作台。
"""

from __future__ import annotations

import streamlit as st

from medical_rag.ui.styles import brand_html, inject_app_styles
from medical_rag.ui.credentials import (
    Credentials,
    credentials,
    storage_file,
    write_credentials,
)
from medical_rag.ui.chat_page import main

st.set_page_config(
    page_title="HyMed-Agent",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化会话状态
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'admin' not in st.session_state:
    st.session_state.admin = False
if 'usname' not in st.session_state:
    st.session_state.usname = ""


def login_page() -> None:
    """渲染登录表单并校验。"""
    with st.form("login_form"):
        username = st.text_input("用户名", value="")
        password = st.text_input("密码", value="", type="password")
        submit = st.form_submit_button("登录", type="primary")

        if submit:
            user_cred = credentials.get(username)
            if user_cred and user_cred.password == password:
                st.success("登录成功！")
                st.session_state.logged_in = True
                st.session_state.admin = user_cred.is_admin
                st.session_state.usname = username
                st.rerun()
            else:
                st.error("用户名或密码错误，请重新输入。")


def register_page() -> None:
    """渲染注册表单并写入凭证文件。"""
    with st.form("register_form"):
        new_username = st.text_input("设置用户名", value="")
        new_password = st.text_input("设置密码", value="", type="password")
        is_admin = False
        register_submit = st.form_submit_button("创建账户", type="primary")

        if register_submit:
            if new_username in credentials:
                st.error("用户名已存在，请使用其他用户名。")
            else:
                # SECURITY: 当前为明文存储，仅供 demo；生产部署必须替换为加盐哈希
                new_user = Credentials(new_username, new_password, is_admin)
                credentials[new_username] = new_user
                write_credentials(storage_file, credentials)
                st.success(f"用户 {new_username} 注册成功！请登录。")
                st.rerun()


if __name__ == "__main__":
    if not st.session_state.logged_in:
        inject_app_styles(login=True)
        st.markdown(brand_html(), unsafe_allow_html=True)
        st.markdown(
            """
            <div class="login-intro">
                <h1>欢迎使用医疗知识问答</h1>
                <p>登录后开始查询医疗知识库与个人健康记忆。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        login_tab, register_tab = st.tabs(["登录", "注册"])
        with login_tab:
            login_page()
        with register_tab:
            register_page()
    else:
        main(st.session_state.admin, st.session_state.usname)
