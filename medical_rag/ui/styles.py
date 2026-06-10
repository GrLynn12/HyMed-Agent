"""Streamlit 页面共享样式。"""

from __future__ import annotations

import streamlit as st


BASE_CSS = """
<style>
:root {
    --ink: #172426;
    --muted: #647476;
    --line: #dfe7e5;
    --surface: #ffffff;
    --surface-soft: #f4f8f7;
    --brand: #147d72;
    --brand-dark: #0c5f57;
    --accent: #d96c4a;
}

.stApp {
    background: #f7faf9;
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: rgba(247, 250, 249, 0.94);
}

[data-testid="stToolbar"] {
    right: 1rem;
}

.block-container {
    max-width: 980px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

h1, h2, h3, p, label, button, input, textarea {
    letter-spacing: 0 !important;
}

h1, h2, h3 {
    color: var(--ink);
}

.app-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.4rem;
}

.brand-mark {
    position: relative;
    width: 38px;
    height: 38px;
    flex: 0 0 38px;
    border-radius: 8px;
    background: var(--brand);
    box-shadow: 0 5px 14px rgba(20, 125, 114, 0.18);
}

.brand-mark::before,
.brand-mark::after {
    content: "";
    position: absolute;
    background: #ffffff;
    border-radius: 2px;
}

.brand-mark::before {
    width: 19px;
    height: 5px;
    left: 9.5px;
    top: 16.5px;
}

.brand-mark::after {
    width: 5px;
    height: 19px;
    left: 16.5px;
    top: 9.5px;
}

.brand-name {
    color: var(--ink);
    font-size: 1.08rem;
    font-weight: 700;
    line-height: 1.2;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 0.16rem;
}

.page-heading {
    margin: 0 0 1.35rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--line);
}

.page-heading h1 {
    margin: 0;
    font-size: 1.65rem;
    line-height: 1.3;
}

.page-heading p {
    margin: 0.35rem 0 0;
    color: var(--muted);
    font-size: 0.92rem;
}

.empty-state {
    padding: 2.4rem 0 1.2rem;
    text-align: center;
}

.empty-state h2 {
    margin: 0;
    font-size: 1.35rem;
}

.empty-state p {
    max-width: 520px;
    margin: 0.55rem auto 0;
    color: var(--muted);
    line-height: 1.7;
}

[data-testid="stSidebar"] {
    background: #edf4f2;
    border-right: 1px solid #d9e5e2;
}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 1.4rem 1rem 1.5rem;
}

[data-testid="stSidebar"] .app-brand {
    margin-bottom: 1.1rem;
}

[data-testid="stSidebar"] hr {
    border-color: #d6e2df;
    margin: 1rem 0;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #5c6d6f;
}

.sidebar-user {
    padding: 0.75rem 0;
    border-top: 1px solid #d6e2df;
    border-bottom: 1px solid #d6e2df;
    margin-bottom: 0.85rem;
}

.sidebar-user strong {
    display: block;
    color: var(--ink);
    font-size: 0.92rem;
}

.sidebar-user span {
    color: var(--muted);
    font-size: 0.76rem;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 7px;
    border: 1px solid #cbd8d5;
    min-height: 2.55rem;
    font-weight: 600;
    box-shadow: none;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border-color: var(--brand);
    color: var(--brand-dark);
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: var(--brand);
    border-color: var(--brand);
    color: #ffffff;
}

.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background: var(--brand-dark);
    border-color: var(--brand-dark);
    color: #ffffff;
}

[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div {
    border-radius: 7px;
    border-color: #cbd8d5;
    background: #ffffff;
}

[data-testid="stChatMessage"] {
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--surface);
    box-shadow: 0 2px 8px rgba(24, 47, 44, 0.035);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #eef7f5;
    border-color: #d3e7e2;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    line-height: 1.72;
}

[data-testid="stChatInput"] {
    border: 1px solid #cbd8d5;
    border-radius: 9px;
    background: #ffffff;
    box-shadow: 0 8px 24px rgba(24, 47, 44, 0.08);
}

[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 7px;
    background: rgba(255, 255, 255, 0.68);
}

[data-testid="stAlert"] {
    border-radius: 7px;
}

@media (max-width: 700px) {
    .block-container {
        padding-top: 1.1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .page-heading h1 {
        font-size: 1.35rem;
    }

    [data-testid="stChatMessage"] {
        padding: 0.85rem;
    }
}
</style>
"""


LOGIN_CSS = """
<style>
[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 520px;
    padding-top: 7vh;
}

.login-intro {
    margin-bottom: 1.5rem;
}

.login-intro h1 {
    margin: 1rem 0 0.35rem;
    font-size: 1.75rem;
}

.login-intro p {
    color: var(--muted);
    line-height: 1.65;
    margin: 0;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 1px solid var(--line);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    height: 3rem;
    padding: 0;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--brand);
}

[data-testid="stForm"] {
    margin-top: 1rem;
    padding: 1.4rem;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--surface);
    box-shadow: 0 10px 30px rgba(24, 47, 44, 0.055);
}

[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    margin-top: 0.4rem;
}

@media (max-width: 700px) {
    .block-container {
        padding-top: 3rem;
    }

    [data-testid="stForm"] {
        padding: 1rem;
    }
}
</style>
"""


def inject_app_styles(login: bool = False) -> None:
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    if login:
        st.markdown(LOGIN_CSS, unsafe_allow_html=True)


def brand_html(compact: bool = False) -> str:
    subtitle = "Hybrid RAG 医疗知识助手" if not compact else "医疗知识助手"
    return f"""
    <div class="app-brand">
        <div class="brand-mark" aria-hidden="true"></div>
        <div>
            <div class="brand-name">HyMed-Agent</div>
            <div class="brand-subtitle">{subtitle}</div>
        </div>
    </div>
    """
