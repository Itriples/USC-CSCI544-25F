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



# RAG Truth Retrieval
1. Core: simulate the process of: LLM with prompt → RAG retrieval → evidence choose method → get evidence → text generation (with evidence chain) -> Generated text → hallucination detection → evidence-hallucination correlation analysis
2. Dataset: RAGTruth, HotpotQA (with label), MS MARCO (without label)
3. Experiment variable: evidence choose strategy
  - top-k vs MMR
  - cross-encoder
  - rewrite
  - context compress
4. Experiment validation:
  - Whether evidence choose strategy choose correct evidence?
      -	Recall of gold evidence (chosen rate of gold evidence over total gold evidence)
      -	Precision of gold evidence (chosen rate of gold evidence over total chosen evidence)
      -	Baseline: Recall@k, nDCG@k, MRR
  - Is chosen evidence consist with data? 
      - Evidence faithfulness: 
      - Supporting facts hit rate (joint F1) with labeled dataset, RAGAS faithfulness / answer relevancy / context precision with unlabeled dataset
  - Under SAME MODEL and PROMPT, how chosen evidence affect the generated text?
      -	End to end precision: EM / F1 (directly compare generation and ground truth?)
  - Efficiency of different strategy:
      -	End to end delay
      -	Context token
  - Hallucination Impact Analysis 
      -	General hallucination rate statistics
      -	Hallucination types distribution of RAGTruth
      -	Evidence quality vs Hallucination
5. Experiment Setup:
  -	Fixed LLM model, prompt template
  -	Building up vector database
  -	Programming evidence choosing algorithm
  -	Composing different choosing method
       -	Only top-k
       - 	Only MMR
       -	Only cross-encoder
       -	Top-k + compression
       -	MMR + cross-encoder
  -	For each of above strategy, measure:
       -	General: Recall@k, MRR, nDCG
       -	For RAGTruth, HotpotQA: supporting facts hit rate
       -	For MS MARCO: RAGAS
       -	General: EM/F1/ROUGE
       -	Delay
  -	Hallucination detection pipeline 
      基于提示的GPT判断 + RAGTruth标签对比
