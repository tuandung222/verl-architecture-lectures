---
sidebar_position: 4
sidebar_label: "Bài 3: Toán học của ReMax"
---

# Bài 3: Toán học của ReMax (Greedy Baseline RLHF)

Giải pháp **ReMax** (được giới thiệu trong bài báo *"ReMax: A Simple, Effective, and Efficient Alternative to PPO"*) là một bước đi đột phá để đơn giản hóa quá trình tối ưu hóa chính sách LLM mà không cần sử dụng mạng Critic ($V_\phi$). 

Bài viết này đi sâu vào phân tích cơ sở toán học của ReMax, chứng minh hiệu quả giảm phương sai của Greedy Baseline và trình bày mã giả chi tiết.

---

## 1. Cơ sở lý thuyết của Greedy Baseline

Mục tiêu tối ưu hóa chính sách RL là tối đa hóa điểm thưởng kỳ vọng:

$$\max_\theta \mathbb{E}_{y \sim \pi_\theta(y | q)} \left[ R(q, y) \right]$$

Để tính toán gradient chính sách một cách ổn định, chúng ta sử dụng toán tử trừ đi baseline để giảm phương sai:

$$g = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(y^s | q) \left( R(q, y^s) - V(q) \right) \right]$$

Trong đó $y^s$ là mẫu được lấy ngẫu nhiên (sampled response).

Thay vì xây dựng và tối ưu một mạng neural Critic độc lập để ước lượng $V(q)$, ReMax định nghĩa baseline chính là **phần thưởng của chuỗi sinh mẫu theo phương thức tham lam (Greedy decoding)**:

$$V(q) \approx R(q, y^g)$$

Trong đó $y^g = \arg\max_y \pi_\theta(y | q)$ là câu trả lời có xác suất tích lũy cao nhất (không áp dụng ngẫu nhiên nhiệt độ).

---

## 2. Chứng minh toán học về việc giảm phương sai (Variance Reduction)

Tại sao việc trừ đi $R(q, y^g)$ lại giúp giảm phương sai của gradient?

Theo lý thuyết xác suất, phương sai của một biến ngẫu nhiên hiệu số $X - Y$ phụ thuộc vào độ tương quan (correlation) giữa chúng:

$$\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y) - 2\text{Cov}(X, Y)$$

Trong đó:
* $X = R(q, y^s) \nabla_\theta \log \pi_\theta(y^s | q)$ (Gradient khi lấy mẫu ngẫu nhiên).
* $Y = R(q, y^g) \nabla_\theta \log \pi_\theta(y^s | q)$ (Gradient của baseline tham lam).

Vì cả $y^s$ và $y^g$ đều được sinh ra từ cùng một phân phối chính sách hiện tại $\pi_\theta$:
* Khi năng lực mô hình thay đổi (ví dụ: mô hình trở nên thông minh hơn và viết tốt hơn), điểm thưởng của cả mẫu ngẫu nhiên $R(q, y^s)$ và mẫu tham lam $R(q, y^g)$ sẽ đồng thời tăng lên.
* Điều này tạo ra độ tương quan dương cực kỳ mạnh mẽ: $\text{Cov}(X, Y) >> 0$.
* Do đó, hiệu số phương sai $\text{Var}(X - Y)$ sẽ nhỏ hơn rất nhiều so với phương sai gốc $\text{Var}(X)$ của gradient thô không có baseline.

Vì $R(q, y^g)$ hoàn toàn độc lập với việc lấy mẫu ngẫu nhiên cụ thể của $y^s$ tại bước hiện tại, việc trừ đi này giữ cho kỳ vọng gradient không chệch (như đã chứng minh ở Bài 2), đồng thời giảm phương sai hiệu quả như Critic mà không cần tốn tài nguyên chạy mạng Critic.

---

## 3. Mã giả thuật toán ReMax

Dưới đây là mã giả thể hiện cách Driver điều phối vLLM để tính toán Lợi thế ReMax và tối ưu hóa chính sách:

```python
import torch

def compute_remax_advantage(
    sampled_rewards: torch.Tensor,     # Shape: (batch_size,) - Điểm RM của câu trả lời ngẫu nhiên
    greedy_rewards: torch.Tensor,      # Shape: (batch_size,) - Điểm RM của câu trả lời tham lam (Greedy Baseline)
    response_mask: torch.Tensor,       # Shape: (batch_size, seq_len)
) -> torch.Tensor:
    
    # 1. Tính toán lợi thế bằng hiệu số điểm thưởng tương đối
    # A = R(y^s) - R(y^g)
    advantages = sampled_rewards - greedy_rewards # Shape: (batch_size,)
    
    # 2. Phát tán lợi thế cấp độ câu xuống các token của response
    # (Tương tự GRPO, lợi thế được giữ cố định trên toàn bộ token của chuỗi sinh mẫu)
    token_advantages = advantages.unsqueeze(-1) * response_mask
    
    return token_advantages

# --- Chu trình huấn luyện chính của ReMax ở Driver ---
def train_remax_step(driver_engine, prompts, target_rewards):
    # 1. Sinh mẫu ngẫu nhiên (Sampling Rollout)
    sampled_batch = driver_engine.rollout_wg.generate_sequences(prompts, do_sample=True)
    sampled_rewards = driver_engine.reward_wg.compute_scores(sampled_batch)
    
    # 2. Sinh mẫu tham lam (Greedy Rollout)
    greedy_batch = driver_engine.rollout_wg.generate_sequences(prompts, do_sample=False)
    greedy_rewards = driver_engine.reward_wg.compute_scores(greedy_batch)
    
    # 3. Tính Lợi thế ReMax
    response_mask = sampled_batch.batch['response_mask']
    advantages = compute_remax_advantage(sampled_rewards, greedy_rewards, response_mask)
    
    # 4. Cập nhật Actor Model
    sampled_batch.batch['advantages'] = advantages
    driver_engine.actor_wg.update_actor(sampled_batch)
```

## 💡 Liên hệ thực tế mã nguồn verl
Trong file [ray_trainer.py](file:///Users/admin/TuanDung/repos/verl/verl/trainer/ppo/ray_trainer.py) của `verl` (dòng 1315-1342), ReMax được hiện thực hóa bằng cách tạo một bản sao sâu (`deepcopy`) của batch đầu vào, thiết lập cờ `do_sample=False` và gửi yêu cầu sinh mẫu tới vLLM Rollout Worker để thu về `reward_baselines`. Cơ chế này hoạt động mượt mà và tận dụng tối đa tốc độ sinh mẫu của vLLM.
