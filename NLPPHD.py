# -*- coding: utf-8 -*-
"""
Created on Thu May 29 01:14:08 2025

@author: Administrator
"""

# streamlit_app.py
# 英汉双语注释：Streamlit 应用，实现对话分类与流程摘要
# Bilingual Comments: Streamlit app for dialogue act classification and progression summary

import streamlit as st  # Web app framework
import re  # 正则解析模块
import math  # 数学计算模块
import nltk  # 自然语言处理
from nltk.tokenize import word_tokenize  # 分词
from transformers import pipeline  # 摘要管道

# --------- NLTK 资源下载（首次运行） ---------
# 英：Download required NLTK models on first run
# 中：首次运行时下载 NLTK 模型
for pkg in ('punkt', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

# --------- Speech Act Principles ---------
# 英：John Searle's five speech act categories and principles
# 中：约翰·塞尔五大言语行为分类及原理
searle = {
    'Assertives': 'Function: state facts, opinions, or beliefs; Direction: utterance→world; Mental state: belief',
    'Directives': 'Function: attempt to get listener to do something; Direction: world→utterance; Mental state: desire',
    'Commissives': 'Function: commit speaker to future action; Direction: world→utterance; Mental state: intention',
    'Expressives': 'Function: express emotional state; Direction: none; Mental state: emotion',
    'Declarations': 'Function: change institutional reality by utterance; Direction: bidirectional; Mental state: institutional power'
}

# --------- Rule keywords for classification ---------
rule_keywords = {
    'Assertives': ["think","believe","know","is","are","was","have","has","had","true","not convinced"],
    'Directives': ["should","need to","have to","try","let's","do","prepare","test","hold off","check","consider"],
    'Commissives': ["will","shall","plan","going to","i'll","we'll","on the roadmap"],
    'Expressives': ["thanks","thank you","sorry","glad","appreciate","maybe"],
    'Declarations': ["i pronounce","declare","announce","appoint"]
}

# Compute 95% confidence interval
def compute_confidence_interval(count, n_keywords, z=1.96):
    p = count / n_keywords if n_keywords else 0
    se = math.sqrt(p * (1 - p) / n_keywords) if n_keywords else 0
    low, high = max(0, p - z*se), min(1, p + z*se)
    return round(low,2), round(high,2)

# Classify a single utterance
def classify_utterance(utter):
    text = utter.lower()
    tokens = word_tokenize(text)
    best_label, best_score, best_ci = None, -1, (0,0)
    for label, kws in rule_keywords.items():
        count = sum(1 for kw in kws if (" " in kw and kw in text) or (kw in tokens))
        score = count / len(kws)
        ci = compute_confidence_interval(count, len(kws))
        if score > best_score:
            best_label, best_score, best_ci = label, score, ci
    return best_label, round(best_score,2), best_ci

# --------- Streamlit UI ---------
st.set_page_config(page_title="Dialogue Analyzer", layout="wide")
st.title("Dialogue Act Classification & Progression Summary")  # English UI title

# Display Searle principles in sidebar
with st.sidebar.expander("John Searle's Speech Act Principles"):
    for k, v in searle.items():
        st.markdown(f"**{k}**: {v}")

# Text input for dialogue
default_dialogue = (
    "Sam: I think we should integrate the chatbot into the main website.\n"
    "Jamie: I’m not convinced - we’ve had issues with reliability.\n"
    "Sam: True, but the latest version uses a more stable backend.\n"
    "Jamie: Have we tested it under real user conditions?\n"
    "Sam: Not yet. That’s on the roadmap for next month.\n"
    "Jamie: Then maybe we hold off integration until those results are in.\n"
    "Sam: Fair enough. I’ll prepare a status update for the next team meeting."
)

dialogue_input = st.text_area("Enter dialogue here:", value=default_dialogue, height=200)

# Process when user clicks button
if st.button("Analyze Dialogue"):
    # Parse into list
    utterances = []
    for line in dialogue_input.split("\n"):
        if ":" in line:
            spk, utt = line.split(":",1)
            utterances.append((spk.strip(), utt.strip()))
    
    # Task 1: classification
    st.header("Task 1: Dialogue Act Classification")
    cols = st.columns([1,4,2,2])
    cols[0].markdown("**Speaker**")
    cols[1].markdown("**Utterance**")
    cols[2].markdown("**Label**")
    cols[3].markdown("**Confidence CI**")
    for spk, utt in utterances:
        label, score, ci = classify_utterance(utt)
        cols = st.columns([1,4,2,2])
        cols[0].write(spk)
        cols[1].write(utt)
        cols[2].write(label)
        cols[3].write(f"{score} ({ci[0]}, {ci[1]})")
    
    # Task 2: summary using local model
    st.header("Task 2: Dialogue Progression Summary")
    # Lazy load summarizer
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
    full_text = " ".join(f"{spk}: {utt}" for spk, utt in utterances)
    summary = summarizer(full_text, max_length=100, min_length=50, do_sample=False)[0]['summary_text']
    st.success(summary)
