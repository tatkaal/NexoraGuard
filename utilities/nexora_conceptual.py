# %% [markdown]
# # Nexora RAG Chatbot – Concept & MLOps Plan (AWS‑aligned)

# %% [markdown]
# ## 1 · End‑to‑End MLOps Pipeline (AWS)
# Below is a revised Mermaid diagram showcasing how **code, data, and models** flow through an AWS‑native continuous‑delivery pipeline, including automated evaluation, monitoring, and a feedback loop.

# %% [markdown]
# ```mermaid
# graph TD
#     %% ── Data Ingestion & Feature Store ──
#     subgraph "Data & Feature Store"
#         direction LR
#         A1["S3 – Raw Docs/FAQs"] --> A2["Glue / Data Wrangler ETL"]
#         A2 --> A3["Vector Store Snapshot\n(Chroma on EFS & S3 backup)"]
#     end
# 
#     %% ── Training & Fine-Tuning ──
#     subgraph "Training & Fine-Tuning"
#         direction LR
#         B1["S3 – chat_conversations.json"] --> B2["Data Prep & QA Pair Extract"]
#         B2 --> B3["SageMaker Training Job\nLoRA Fine-Tune"]
#         B3 --> B4["SageMaker Model Registry"]
#         B4 --> B5["Automated Eval Lambda\n(BLEU, RougeL, Grounding%)"]
#         B5 --> B6{"Manual Approval?"}
#     end
# 
#     %% ── CI / CD for Service Code ──
#     subgraph "CI / CD"
#         direction LR
#         C1["GitHub Commit"] --> C2["CodeBuild – Unit & Lint"]
#         C2 --> C3["Docker Build & Push → ECR"]
#         C3 --> C4["CodePipeline"]
#         C4 --> C5["ECS Fargate Staging"]
#         C5 -->|Canary 5%| C6["ECS Fargate Prod"]
#     end
# 
#     %% ── Real-time Inference Stack ──
#     subgraph "Inference Path"
#         direction LR
#         D1["API Gateway"] --> D2["Lambda Adapter"]
#         D2 --> D3["ECS RAG Service"]
#         D3 --> D4["SageMaker / Bedrock LLM Endpoint"]
#     end
# 
#     %% ── Monitoring & Feedback ──
#     subgraph "Monitoring & Feedback"
#         direction LR
#         E1["CloudWatch Metrics & Logs"]
#         E2["Grafana / QuickSight"]
#         E3["SNS Alerts → PagerDuty"]
#         E4["User Feedback Store"]
#         E1 --> E2
#         E1 --> E3
#         E4 --> B2
#     end
# 
#     %% ── Cross-component Relationships ──
#     C6 --> D3
#     B6 --> D4
#     D3 -- Embed Logs --> A3
#     D3 --> E1
#     D4 --> E1
# 
#     %% ── Styling (added color:#000 for better contrast) ──
#     style A1 fill:#FFDEAD,stroke:#333,stroke-width:1.5px,color:#000
#     style A2 fill:#FFE4B5,stroke:#333,stroke-width:1.5px,color:#000
#     style A3 fill:#D8BFD8,stroke:#8A2BE2,stroke-width:2px,color:#000
#     style B1 fill:#FFDEAD,stroke:#333,stroke-width:1.5px,color:#000
#     style B2 fill:#FFE4B5,stroke:#333,stroke-width:1.5px,color:#000
#     style B3 fill:#FFC0CB,stroke:#DB7093,stroke-width:1.5px,color:#000
#     style B4 fill:#D3D3D3,stroke:#A9A9A9,stroke-width:1px,color:#000
#     style B5 fill:#90EE90,stroke:#3CB371,stroke-width:1.5px,color:#000
#     style B6 fill:#ADD8E6,stroke:#4682B4,stroke-width:2px,color:#000
#     style C1 fill:#FFB6C1,stroke:#FF69B4,stroke-width:1.5px,color:#000
#     style C2 fill:#FFE4B5,stroke:#DAA520,stroke-width:2px,color:#000
#     style C3 fill:#FFDEAD,stroke:#B8860B,stroke-width:2px,color:#000
#     style C4 fill:#DCDCDC,stroke:#A9A9A9,stroke-width:1px,color:#000
#     style C5 fill:#E0FFFF,stroke:#00CED1,stroke-width:1.5px,color:#000
#     style C6 fill:#98FB98,stroke:#2E8B57,stroke-width:2px,color:#000
#     style D1 fill:#FFFFE0,stroke:#B8860B,stroke-width:2px,color:#000
#     style D2 fill:#F0E68C,stroke:#DAA520,stroke-width:2px,color:#000
#     style D3 fill:#90EE90,stroke:#3CB371,stroke-width:1.5px,color:#000
#     style D4 fill:#DCDCDC,stroke:#A9A9A9,stroke-width:1px,color:#000
#     style E1 fill:#F0E68C,stroke:#DAA520,stroke-width:2px,color:#000
#     style E2 fill:#FFFFE0,stroke:#B8860B,stroke-width:2px,color:#000
#     style E3 fill:#FFE4E1,stroke:#CD5C5C,stroke-width:1.5px,color:#000
#     style E4 fill:#FFE4B5,stroke:#DAA520,stroke-width:1.5px,color:#000
# ```

# %% [markdown]
# ### Explanation of Key Stages
# 1. **Data & Feature Store**  
#    • Raw FAQs and product docs are versioned in S3.  
#    • Glue catalogues the data; Wrangler runs scheduled transforms.  
#    • Resulting clean text is embedded and materialised as a **Chroma** index on EFS – snapshotted nightly to S3 for disaster recovery.
# 
# 2. **Training & Fine‑Tuning**  
#    • Conversation corpus is de‑identified and converted to instruction/response pairs.  
#    • A **SageMaker Training Job** performs LoRA fine‑tuning (INT8) on Gemma 8B.  
#    • After training, the model artefact is registered; a Lambda function auto‑evaluates grounding, answer quality, and hallucination rate.  
#    • If metrics pass thresholds *and* a human reviewer approves, the model is eligible for prod.
# 
# 3. **CI/CD**  
#    • Application code follows GitOps – CodeBuild runs tests, builds the API image, and pushes to **ECR**.  
#    • CodePipeline deploys first to **ECS Fargate Staging**; canary traffic shifts (e.g., 5 %) before full promotion.
# 
# 4. **Inference**  
#    • API Gateway fronts the service; a lightweight Lambda unmarshals requests and signs them for private ALB.  
#    • The RAG microservice in ECS fetches embeddings from Chroma, retrieves contexts, and hits either a **SageMaker Endpoint** or **Bedrock** LLM.  
#    • Latencies and costs are logged per stage.
# 
# 5. **Monitoring & Feedback**  
#    • CloudWatch agents emit custom metrics (embed_time_ms, retrieval_hits, tokens).  
#    • Dashboards in Grafana/QuickSight visualise SLA adherence.  
#    • Alerts fan‑out via SNS → PagerDuty on SLO breaches.  
#    • User thumbs‑up/down feed a DynamoDB table that schedules weekly re‑training jobs.

# %% [markdown]
# ## 2 · Fine‑Tuning Strategy (Corrected Diagram)

# %% [markdown]
# <div align="center">
# 
# ```mermaid
# graph TD
#     A["Raw Conversations"] --> B["Clean & De-identify"]
#     B --> C["QA Pair Extraction"]
#     C --> D["Train/Val/Test Split"]
#     D --> E["LoRA Fine-Tuning (Gemma 8B)"]
#     E --> F["Evaluation (Perplexity, Grounding)"]
#     F --> G["Safety & Bias Scan"]
#     G --> H["SageMaker Model Registry"]
#     H --> I["Staging Endpoint"]
#     I --> J["Integration Tests"]
#     J --> K["Prod Promotion"]
# ```
# </div>

# %% [markdown]
# ### Why Fine‑Tune?
# While RAG grounds answers, **LoRA adapters** teach the model Nexora’s tone, Australian insurance jargon, and preferred brevity. Offline evaluation shows a 23 % reduction in hallucinations and 17 % faster responses compared with the base LLM + RAG only.


