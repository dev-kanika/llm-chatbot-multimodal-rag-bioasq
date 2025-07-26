# 🧠 Multi-Approach LLM Chatbot for Biomedical Q&A

A powerful **multimodal chatbot** system trained on biomedical data using five distinct large language modeling techniques. This project demonstrates and compares various **retrieval and generative models** from causal transformers to RAG pipelines with query rewriting—to build a robust, domain-safe question-answering assistant.

---

## 📂 Dataset

All models are built using data from the **BioASQ dataset**, retrieved from HuggingFace:

```python
df = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
df = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/test.parquet/part.0.parquet")
```
---

## 🔧 Approaches Implemented

| Model                        | Description                                                              | Techniques Used                               |
|-----------------------------|--------------------------------------------------------------------------|------------------------------------------------|
| **1. Causal Transformer**   | Language model with pretrained embeddings and **relative position encoding** | PyTorch, Positional Embeddings                 |
| **2. RAG (no rewrite)**     | **Retriever-augmented generation** with top-10 chunk retrieval using FAISS | LLaMA / Gemini, FAISS, LangChain              |
| **3. Finetuned LLM**        | PEFT-tuned **TinyLLaMA** via QLoRA                                       | PEFT, QLoRA, HuggingFace Transformers          |
| **4. RAG + Query Rewrite**  | RAG with **Gemma-based query rewriting** for better context relevance    | Gemma, DSPy, LangChain                         |
| **5. Prompt Engineering**   | Few-Shot, Chain-of-Thought, DSP prompting on pretrained LLM              | Prompt Engineering, CoT, DSP                   |

> _All models reject out-of-context queries such as "What is the effect of tariffs on the economy?"_

---

## 🛠️ Tech Stack

- **LLMs:** LLaMA, Gemma, GPT-2, TinyLLaMA  
- **RAG Toolkits:** FAISS, LangChain, DSPy  
- **Tuning:** QLoRA, LoRA (PEFT)  
- **Prompting:** Few-Shot, Chain-of-Thought (CoT)  
- **Evaluation:** ROUGE, BERTScore, SHAP, MAP, MRR  

---

## 📈 Model Performance

| Model                   | ROUGE-L | BERT-F1 | MAP (RAG) | MRR (RAG) |  AUC |
|-------------------------|---------|---------|-----------|-----------|------|
| Causal Transformer      | 0.60    | 0.95    | —         | —         | 0.80 |
| RAG (no rewrite)        | 0.48    | 0.79    | 0.61      | 0.66      | 0.87 |
| PEFT (TinyLLaMA)        | 0.26    | 0.86    | —         | —         | —    |
| RAG + Gemma Rewrite     | 0.74    | 0.79    | 0.65      | 0.69      | —    |
| Prompt Engineering      | 0.42    | 0.94    | —         | —         | 0.87 |

> 🔍 _Evaluation performed using ROUGE-L, BERT-F1, and retrieval metrics (MAP, MRR)._

---

## 📌 My Contribution

- Designed and implemented **all five modeling pipelines** from scratch  
- Integrated **retrieval and generation workflows** using LangChain + FAISS  
- Applied **PEFT tuning** and advanced prompt engineering  
- Evaluated using **ROUGE-L**, **BERTScore**, **MAP**, and **MRR** 
- Ensured **safety filters** for out-of-domain queries  

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/llm-chatbot-multimodal-rag-bioasq.git
pip install -r requirements.txt
```
---

## 📊 Future Improvements

- Integrate with Streamlit or Gradio UI
- Improve generation with Hybrid RAG + PEFT
- Expand with BioGPT or PubMedBERT for domain enhancement

---

## 📎 License

This project is for academic purposes only. For licensing details, see LICENSE.md.


