---
sidebar_position: 1
sidebar_label: "Lộ trình Case Studies"
---

# Lộ trình Case Studies: ART trong thực chiến

Phần này trình bày **5 tình huống thực tế** mà ART đã được sử dụng để giải quyết các bài toán agentic RL cụ thể. Mỗi case study đi kèm mã nguồn tối giản, sơ đồ kiến trúc, và phân tích tại sao chọn backend/feature này.

Các case study **không theo thứ tự bắt buộc**; bạn có thể đọc theo bất kỳ thứ tự nào tùy quan tâm.

```mermaid
graph TD
    A[Bạn muốn làm gì?] --> B[Email agent tự động]
    A --> C[Game có reward rõ ràng]
    A --> D[Agent dùng MCP server]
    A --> E[Tích hợp LangGraph]
    A --> F[Zero-infra, serverless]

    B --> C1[Case 1: ART·E Email Agent]
    C --> C2[Case 2: 2048 GRPO]
    D --> C3[Case 3: MCP·RL]
    E --> C4[Case 4: LangGraph Integration]
    F --> C5[Case 5: W&B Serverless RL]

    style A fill:#1d4ed8,stroke:#3b82f6,color:#fff
    style C1 fill:#065f46,stroke:#10b981,color:#fff
    style C2 fill:#065f46,stroke:#10b981,color:#fff
    style C3 fill:#065f46,stroke:#10b981,color:#fff
    style C4 fill:#065f46,stroke:#10b981,color:#fff
    style C5 fill:#065f46,stroke:#10b981,color:#fff
```

---

## Tổng quan 5 case studies

| # | Tên | Tình huống | Backend | Tính năng nổi bật |
| --- | --- | --- | --- | --- |
| 1 | ART·E Email Agent | Agent soạn email theo lịch sử hội thoại | LocalBackend (Unsloth + vLLM) | RULER judge, multi-turn tool use, Slack logging |
| 2 | 2048 GRPO | Game 2048 với reward dựa trên score | LocalBackend (Qwen 2.5 3B) | 18 simultaneous games, ruler_score_group, S3 checkpoint |
| 3 | MCP·RL | Train model dùng MCP servers (filesystem, git) | LocalBackend | OpenAI proxy + HTTPX, MCP tool calling |
| 4 | LangGraph Integration | Tích hợp ART với LangGraph agent | LocalBackend | `wrap_rollout`, `init_chat_model`, conversation reconstruction |
| 5 | W&B Serverless RL | Training zero-infra, không cần GPU local | ServerlessBackend | W&B Serverless RL, 2048 benchmark, no GPU setup |

---

## Sơ đồ so sánh backend sử dụng

```mermaid
graph LR
    C1[Case 1: ART·E] --> LB1[LocalBackend + Unsloth]
    C2[Case 2: 2048] --> LB2[LocalBackend + Qwen 2.5 3B]
    C3[Case 3: MCP·RL] --> LB3[LocalBackend + Unsloth]
    C4[Case 4: LangGraph] --> LB4[LocalBackend]
    C5[Case 5: Serverless] --> SB[ServerlessBackend - W&B]

    LB1 --> V1[vLLM + CUDA]
    LB2 --> V2[vLLM + CUDA]
    LB3 --> V3[vLLM + CUDA]
    LB4 --> V4[vLLM + CUDA]
    SB --> W[B Weave Serverless]
```

* **Case 1-4** chạy local: yêu cầu GPU NVIDIA + vLLM + Unsloth (hoặc TRL/Megatron). Phù hợp khi bạn cần kiểm soát hoàn toàn.
* **Case 5** chạy serverless: không cần GPU; W&B thuê GPU từ cloud provider. Phù hợp khi bạn muốn zero-setup hoặc benchmark nhanh.

---

## Bài học rút ra từ mỗi case

* **Case 1 (ART·E)**: minh họa `ruler_score_group` + RULER judge hoạt động tốt khi task subjective, không có ground truth.
* **Case 2 (2048)**: minh họa pattern "K rollout cùng seed, tính advantage, train step" cổ điển của GRPO. Dễ hiểu nhất cho người mới.
* **Case 3 (MCP·RL)**: minh họa HTTPX patching + `auto_trajectory` capture đa agent trong suốt.
* **Case 4 (LangGraph)**: minh họa cách wrap framework khác (LangGraph) vào ART bằng logging + reconstruction.
* **Case 5 (Serverless)**: minh họa khi nào nên dùng cloud-hosted RL thay vì tự vận hành.

---

## Khi nào chọn case nào?

```mermaid
graph TD
    Q1{Bạn có GPU local không?}
    Q1 -- Có --> Q2{Reward có sẵn?}
    Q1 -- Không --> C5[Case 5: Serverless]

    Q2 -- Có --> Q3{Game hay task mở?}
    Q2 -- Không --> Q4{Framework sẵn có?}

    Q3 -- Game --> C2[Case 2: 2048]
    Q3 -- Task mở --> C1[Case 1: ART·E]

    Q4 -- LangGraph --> C4[Case 4]
    Q4 -- MCP --> C3[Case 3]
    Q4 -- Code thuần --> C1[Case 1]
```

---

Bắt đầu với [Case 1: ART·E Email Agent](case_1_art_e_email_agent) - case study nổi tiếng nhất của OpenPipe.
