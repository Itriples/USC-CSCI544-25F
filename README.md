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



# Multi agent research
Title:
Hierarchical Multi-Agent QA with Specialized Roles and a Verifier for Attribution Reliability

1. Motivation

Large Language Models (LLMs) often hallucinate or provide unsupported claims in complex QA tasks. While multi-agent systems have shown promise, they still lack attribution reliability — the ability to ground every answer in verifiable evidence. Recent industry systems (e.g., Anthropic’s Research Assistant with a Citation Agent) highlight this gap, but systematic academic evaluation is missing.

Our project aims to investigate whether combining (a) role specialization (Searcher, Analyst, Writer) and (b) an independent Verifier/Citation Agent under a centralized Orchestrator can significantly improve accuracy–attribution–cost trade-offs in QA tasks.

2. Literature Review

Anthropic (2025). How we built a multi-agent research system.
[Anthropic Blog] – 描述 Lead Researcher + Citation Agent 架构，强调证据溯源和宽度优先研究查询。

Jin et al. (2025). Talk Hierarchically, Act Structurally: Structured Communication Protocols for Multi-Agent LLM Systems. arXiv:2502.11098.
提出 structured communication + hierarchical refinement，有效减少协作幻觉。

Zhang et al. (2025). MACT: Multi-agent Cooperative Tuning for Complex Reasoning with LLMs. NAACL 2025.
用多代理协作处理表格问答，证明 planner–executor–tool 结合的有效性。

Li et al. (2023). CAMEL: Communicative Agents for “Mind” Exploration. ICLR 2023.
经典工作，提出多代理角色扮演框架。

Wu et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. Microsoft Research.
多代理框架，广泛用于 orchestrator–worker 实现。

Sun et al. (2024). MegaAgent: Scaling LLM-based Multi-Agent Collaboration without Predefined SOPs. arXiv:2408.09955.
探索上百代理的自治与动态生/杀 agent 管理。

Zhou et al. (2024). CollabEval: Enhancing LLM-as-a-Judge via Multi-Agent Collaboration. Amazon Science.
提出三阶段评估 (初评–讨论–定夺)，提升一致性和鲁棒性。

Rashkin et al. (2021). ASQA: Factually Consistent Long-Form Question Answering. NAACL 2021.
提出 Attribution-based QA 数据集，适合评测 citation precision/recall。

Kamalloo et al. (2023). Evaluating Attributed QA: Can LLMs Reason with Cited Evidence? arXiv:2310.12848.
系统提出 Attribution QA 指标（Citation F1、Attribution Recall）。

Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. ACL 2023.
用异质采样检测 LLM 幻觉，可作为 Verifier 评价参考。

Chen et al. (2023). QAFactEval: Improved QA-based Factual Consistency Evaluation for Summarization. EMNLP 2023.
用于句子-证据蕴含度检验，常用于 QA/总结验证。

Gupta et al. (2022). FEVER: A Large-scale Dataset for Fact Extraction and Verification. NAACL 2022.
事实核查数据集，标注支持/反驳/无法判断与证据句。

Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.
多跳问答数据集，含 supporting facts，可用于 attribution 评测。

3. Proposed Method

Architecture:

Orchestrator: decomposes task, assigns roles.

Sub-agents:

Searcher retrieves evidence,

Analyst links evidence to sub-questions,

Writer synthesizes final answer with inline citations.

Verifier/Citation Agent: checks if every statement is properly supported (using AIS/QAFactEval), rejects unsupported outputs → triggers rework.

Structured Protocol: every intermediate/final answer must include evidence slots (document ID + passage).

4. Experiments

Datasets:

HotpotQA (multi-hop QA with supporting facts)

FEVER (fact verification with labeled evidence)

(Optional) ASQA / ELI5-subset (long-form QA with attribution labels)

Metrics:

Accuracy: EM/F1 (HotpotQA), Label Accuracy (FEVER)

Attribution: AIS, Citation Precision/Recall/F1

Cost: tokens, #rounds, latency

Baselines:

Single-agent RAG/ReAct

Multi-agent (no specialization)

Multi-agent (specialization, no verifier)

5. Plan (Milestones)

Week 1–2: Literature survey + single-agent baseline (RAG/ReAct).

Week 3–4: Implement Orchestrator + specialized Sub-agents; run HotpotQA baseline.

Week 5–6: Add Verifier/Citation Agent + structured protocol; full evaluation on HotpotQA + FEVER.

Week 7–8: Ablation (remove verifier, remove specialization, remove evidence slots); analyze attribution–accuracy–cost trade-off; finalize report.

6. Expected Contributions

Empirical evidence that Verifier + specialization improves attribution reliability.

Quantitative analysis of accuracy–attribution–cost trade-offs.

Ablation showing which components contribute most (roles, evidence slots, verifier).

7. Datasets

Potential Usecase 1: FEVER (Thorne et al., 2018; Gupta et al., 2022 revision)
- Task: Fact verification — given a claim, classify as SUPPORTED, REFUTED, or NOT ENOUGH INFO using Wikipedia evidence.
- Why relevant: Directly evaluates attribution reliability since evidence must justify label.
- Agentic Example:
  - Claim: “The Eiffel Tower is taller than the Empire State Building.”
  - Workflow:
	1.	Searcher retrieves Wikipedia entries for Eiffel Tower and Empire State Building.
	2.	Analyst extracts relevant heights (324m vs 381m).
	3.	Writer produces: “The claim is REFUTED: Eiffel Tower (324m) < Empire State Building (381m) [Wiki:Eiffel Tower; Wiki:Empire State Building].”
	4.	Verifier confirms that numeric comparison is present in retrieved evidence.

Public Datasources to Use
- Wikipedia dumps (for HotpotQA/FEVER, readily available via Hugging Face Datasets or ElasticSearch-backed retrieval).
- HotpotQA supporting facts (via Hugging Face Datasets: hotpot_qa).
- FEVER dataset (via Hugging Face Datasets: fever).


**Potential Usecase 2: ASQA (Attribution QA) (Rashkin et al., 2021)**
- Task: Long-form QA with attribution labels — requires generating comprehensive answers with fine-grained citations.
- Why relevant: Tests if the multi-agent system can maintain coverage + reliability at scale in long-form outputs.
- Agentic Example:
  - Question: “Explain the contributions of Ada Lovelace to computing.”
  - Workflow:
	1.	Orchestrator splits into facets: early life, collaboration with Babbage, significance of notes on Analytical Engine.
	2.	Searcher retrieves multiple relevant passages.
	3.	Analyst links evidence to each sub-facet.
	4.	Writer synthesizes multi-paragraph answer with inline citations after each claim.
	5.	Verifier ensures every factual statement (e.g., “first computer programmer,” “notes on Analytical Engine”) is attributed.

Public Datasources to Use:
- ASQA dataset (via Hugging Face Datasets: asqa).
- Optional: QAFactEval synthetic pairs (Chen et al., 2023) to test verifier robustness.

 
**Potential Usecase 3: Multi-Agent Market Research**

1. Competitor Data (Scraper Agent)
	•	Crunchbase Open Data Map (https://data.crunchbase.com/)
	•	Company funding, investors, industry tags.
	•	CB Insights (limited public reports) (https://www.cbinsights.com/research/)
	•	Industry trend snapshots.
	•	Alternative: Kaggle Startup Datasets (e.g., startups, companies, startup-investments) — cleaned competitor/funding info.
	•	SEC EDGAR Filings (https://www.sec.gov/edgar.shtml)
	•	Public company financials & competitor positioning.

2. Market Trend Summarization (Analyst Agent)
	•	World Bank Data (https://data.worldbank.org/)
	•	Macroeconomic and sectoral data.
	•	OECD Data Explorer (https://data-explorer.oecd.org/)
	•	Industry performance & policy trends.
	•	Statista Public Reports (free subsets) (https://www.statista.com/)
	•	Consumer adoption, market penetration (limited free tier).
	•	Google Trends API (pytrends)
	•	Consumer interest signals, seasonal product demand.

3. SWOT Analysis Generation (Writer Agent)
	•	Builds on above sources but benefits from:
	•	Kaggle “Business/Industry Reports” Datasets
	•	E.g., retail, fintech, SaaS growth trends.
	•	IBISWorld Summaries (free previews)
	•	Sector strengths/weaknesses at macro level.
	•	PitchBook public blog insights
	•	Emerging market opportunities.

4. Validation (Verifier Agent)
	•	News & Fact-Checking Sources:
	•	GDELT Project (https://www.gdeltproject.org/) — global news events, entity mentions.
	•	NewsAPI (https://newsapi.org/) — business/financial news feeds.
	•	Kaggle: Fake News / Fact-checking Datasets (for testing citation verification).
	•	Benchmark Datasets for Factual QA:
	•	FEVER (claim verification).
	•	QAFactEval (evidence-supported factuality checking).

**Example Workflow with Public Data**

Use Case: A startup in EV charging infrastructure wants competitor + trend insights.
1.	Scraper Agent → pulls competitor data from Crunchbase API (funding rounds of EV charging companies).
2.	Analyst Agent → summarizes growth signals using World Bank EV adoption stats + Google Trends (“EV charging station near me”).
3.	Writer Agent → generates SWOT (Strength: rising demand; Weakness: capital-intensive infra; Opportunity: policy subsidies; Threat: Tesla Supercharger dominance).
4.	Verifier Agent → cross-checks claims against news feeds (GDELT/NewsAPI) and funding numbers from Crunchbase.
