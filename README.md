USC-CSCI544-25F

# 1. LLMs for Time Series Forecasting
## Idea
Use **LLMs** to improve **time series forecasting (TSF)** tasks.

## Related Papers
- [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)
- [CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning](https://arxiv.org/abs/2403.07300)

## Datasets
- **ETT** (Electricity Transformer Temperature)
- **Electricity**
- **Traffic** <br>
See datasets [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)



# 2. Multi-Agent Market Researcher (Applied)
- Agents: one scrapes competitor data, one summarizes market trends, one generates SWOT analysis, one validates findings.
- Use case: Automates consulting-style market research for startups or SMEs.

# 3. 	Recommendation System with Text + Metadata (Applied)
- Combine embeddings from text (e.g., product reviews) with structured data (ratings, price) for hybrid recommendations.
- Great e-commerce angle.

# 4.  Faithful & Source-Grounded Summarization (Research)
- Build a summarization model that not only generates concise summaries but also highlights supporting evidence spans in the source documents.
- Novelty: Tackles the hallucination problem in LLM summarization.
- Reference:
  - Maynez, J., et al. (2020). On Faithfulness and Factuality in Abstractive Summarization. ACL. 📄
	- Ladhak, F., et al. (2022). Faithful Summarization with Attribution. ACL. 📄

# 5. Explainable Toxicity Detection (Research)
- Train a toxicity classifier that not only predicts labels but also provides human-interpretable explanations (e.g., highlighting offensive spans).
- Novelty: Bridges fairness + explainability in NLP.
- Reference:
  - Sap, M., et al. (2019). The Risk of Racial Bias in Hate Speech Detection. ACL. 📄
	- Mathew, B., et al. (2021). Hatexplain: A Benchmark Dataset for Explainable Hate Speech Detection. AAAI


## 🔹 Application-Oriented Projects

### 1. Multilingual Sentiment Analysis Extension
**Motivation:**  
Sentiment analysis is a well-studied NLP task, but most benchmarks focus on English. Extending models to low-resource or multilingual settings is both practical and impactful.  

**Literature Review:**  
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.  
- Mozafari et al. (2019). *A BERT-based Transfer Learning Approach for Hate Speech Detection*. NLPCC.

### 2. Information Extraction in Legal Documents
**Motivation:**  
Legal documents are dense and domain-specific, making them ideal for testing Named Entity Recognition (NER) and Relation Extraction (RE). Applying NLP models here supports real-world applications like contract review and case law analysis.  

**Literature Review:**  
- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School*. Findings of EMNLP.  
- Beltagy et al. (2019). *SciBERT: A Pretrained Language Model for Scientific Text*. EMNLP.

## 🔹 Research-Oriented Projects

### 3. Prompt Robustness in Adversarial Settings
**Motivation:**  
Prompt-based learning is widely adopted, but prompts are fragile under adversarial modifications (e.g., typos, paraphrases). This project studies how robust different prompting methods are under noisy conditions.  

**Literature Review:**  
- Jin et al. (2020). *Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment*. AAAI.  
- Zhao et al. (2021). *Calibrate Before Use: Improving Few-Shot Performance of Language Models*. ICML.

### 4. Cross-Modal Knowledge Transfer (Vision → NLP)
**Motivation:**  
Techniques from computer vision (e.g., Vision Transformer pooling) may improve long-context NLP tasks such as document classification or summarization. Exploring such cross-modal transfer can reveal novel design insights.  

**Literature Review:**  
- Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)*. ICLR.  
- Beltagy et al. (2020). *Longformer: The Long-Document Transformer*. arXiv.
