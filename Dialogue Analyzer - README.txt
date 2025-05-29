
Overview
This Streamlit application performs dialogue act classification (Task 1) and generates narrative summaries (Task 2) from conversational text input. The tool implements John Searle's speech act theory for classification and uses a transformer-based language model for abstractive summarization.

Installation & Usage
Prerequisites:

Python 3.7+

Required packages: streamlit transformers nltk

Installation:

bash
pip install streamlit transformers nltk
Running the app:

bash
streamlit run NLPPHD.py
Usage:

Enter dialogue in the text area (default example provided)

Click "Analyze Dialogue" button

View classification results and generated summary

Implementation Details
Task 1: Dialogue Act Classification
Implements rule-based classification using Searle's five speech act categories

Uses keyword matching with confidence interval calculation

Displays results in an interactive table format

Task 2: Progression Summary
Uses Hugging Face's sshleifer/distilbart-cnn-12-6 model

140M parameter distilled version of BART transformer

Fine-tuned specifically for summarization tasks

Runs locally - no API keys required

Generates abstractive (not extractive) summaries

Model Specifications
Model Type: DistilBART (Transformer-based seq2seq)

Parameters: 140 million

Training: Fine-tuned on CNN/Daily Mail dataset

Capabilities:

Understands conversational dynamics

Maintains logical flow in summaries

Handles speaker interactions

Generates human-readable narratives

Technical Notes
First run will download NLTK data (~1.8MB) and the language model (~1.5GB)

All processing occurs locally - no external API calls

Summary length adjustable via code parameters