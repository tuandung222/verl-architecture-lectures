---
sidebar_position: 1
sidebar_label: "Lộ trình Lý thuyết"
---

# Lộ trình Lý thuyết & Toán học của verl

Chào mừng bạn đến với chương **Đào sâu Lý thuyết, Toán học và Mã giả (Math & Theory Deep Dive)**.

Để làm chủ hoàn toàn các hệ thống căn chỉnh LLM phân tán, chúng ta phải vượt qua lớp vỏ bọc giao diện và mã nguồn để đi vào gốc rễ lý thuyết toán học của các thuật toán. Phần này sẽ cung cấp các chứng minh toán học đầy đủ, phân tích cơ chế giảm phương sai, thuật toán thích ứng và mã giả đi kèm tham chiếu mã nguồn của `verl`.

---

```mermaid
graph TD
    T1[Bài 1: Toán học của PPO & GAE] --> T2[Bài 2: Toán học của GRPO & Baseline Nhóm]
    T2 --> T3[Bài 3: Toán học của ReMax & Greedy Baseline]
    T3 --> T4[Bài 4: Lý thuyết về KL Divergence & Adaptive Control]
    T4 --> T5[Bài 5: Đại số Resharding & Độ phức tạp Giao tiếp]

    style T1 fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    style T2 fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff
    style T3 fill:#92400e,stroke:#f59e0b,stroke-width:2px,color:#fff
    style T4 fill:#9d174d,stroke:#ec4899,stroke-width:2px,color:#fff
    style T5 fill:#5b21b6,stroke:#8b5cf6,stroke-width:2px,color:#fff
```

---

## Nội dung các bài giảng Toán học chuyên sâu

1. **[Bài 1: Toán học của PPO (GAE, Clipped Objective & Value Estimator)](theory_1_ppo_math)**
   * Bản chất toán học của Clipped Surrogate Loss, cơ chế giảm phương sai của GAE và mã giả.
2. **[Bài 2: Toán học của GRPO (Ước lượng Baseline Nhóm & Giảm phương sai)](theory_2_grpo_math)**
   * Chứng minh toán học ước lượng không chệch (unbiased) của GRPO và cơ chế loại bỏ Critic.
3. **[Bài 3: Toán học của ReMax (Greedy Baseline RLHF)](theory_3_remax_math)**
   * Cơ sở toán học của việc sử dụng Greedy Rollout làm baseline giảm phương sai.
4. **[Bài 4: Cơ sở toán học của KL Divergence & Bộ điều phối Adaptive KL](theory_4_kl_divergence)**
   * Định nghĩa thông tin của KL Penalty, Trust Region bounds, và giải thuật cập nhật $\beta$ động.
5. **[Bài 5: Đại số phân mảnh (Resharding Algebra) & Độ phức tạp Giao tiếp](theory_5_resharding_algebra)**
   * Đại số ma trận chuyển đổi FSDP sang TP và độ phức tạp truyền thông NCCL/All-to-All.
