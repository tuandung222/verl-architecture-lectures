---
sidebar_position: 3
sidebar_label: "Bài 2: Toán học của GRPO & Baseline Nhóm"
---

# Bài 2: Toán học của GRPO (Ước lượng Baseline Nhóm & Giảm phương sai)

Thuật toán **GRPO (Group Relative Policy Optimization)** do DeepSeek phát triển là một bước cải tiến lớn giúp loại bỏ mạng Critic ($V_\phi$) trong huấn luyện RLHF. Bài viết này chứng minh toán học tại sao việc lấy trung bình nhóm làm baseline vẫn đảm bảo ước lượng gradient không chệch (unbiased) và cách tính toán lợi thế tương đối giúp giảm phương sai.

---

## 1. Chứng minh toán học: Ước lượng Gradient không chệch (Unbiased Estimator)

Trong học tăng cường chính sách tiêu chuẩn, gradient của hàm mục tiêu được ước lượng qua:

$$g = \mathbb{E}_{y \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(y | q) (R(q, y) - V(q)) \right]$$

Trong đó $V(q)$ là một hàm baseline bất kỳ độc lập với câu trả lời $y$ hiện tại (thường là mạng Critic $V_\phi(q)$).

GRPO thay thế $V(q)$ bằng giá trị trung bình mẫu của một nhóm $G$ câu trả lời sinh ra từ cùng một prompt $q$:

$$V(q) \approx \bar{R} = \frac{1}{G} \sum_{j=1}^G R(q, y_j)$$

### Chứng minh toán học:
Chúng ta cần chứng minh rằng việc trừ đi trung bình nhóm $\bar{R}$ không làm thay đổi kỳ vọng của gradient chính sách.

Xét kỳ vọng của thành phần trừ đi baseline nhóm:

$$\mathbb{E}_{y_1..y_G \sim \pi_\theta} \left[ \sum_{i=1}^G \nabla_\theta \log \pi_\theta(y_i | q) \left( R(q, y_i) - \frac{1}{G} \sum_{j=1}^G R(q, y_j) \right) \right]$$

Khai triển tổng bên trong:

$$= \mathbb{E} \left[ \sum_{i=1}^G \nabla_\theta \log \pi_\theta(y_i | q) R(q, y_i) \right] - \mathbb{E} \left[ \sum_{i=1}^G \nabla_\theta \log \pi_\theta(y_i | q) \left( \frac{1}{G} \sum_{j=1}^G R(q, y_j) \right) \right]$$

Xét thành phần thứ hai (phần trừ đi baseline):

$$\sum_{i=1}^G \nabla_\theta \log \pi_\theta(y_i | q) \left( \frac{1}{G} \sum_{j=1}^G R(q, y_j) \right) = \frac{1}{G} \sum_{i=1}^G \sum_{j=1}^G \nabla_\theta \log \pi_\theta(y_i | q) R(q, y_j)$$

Tách tổng khi $i \neq j$ và $i = j$:
Vì các câu trả lời $y_i$ và $y_j$ được lấy mẫu độc lập (IID) từ $\pi_\theta$, khi $i \neq j$, kỳ vọng của tích bằng tích các kỳ vọng:

$$\mathbb{E} \left[ \nabla_\theta \log \pi_\theta(y_i | q) R(q, y_j) \right] = \mathbb{E}_{y_i} \left[ \nabla_\theta \log \pi_\theta(y_i | q) \right] \cdot \mathbb{E}_{y_j} \left[ R(q, y_j) \right]$$

Ta biết rằng kỳ vọng của gradient log xác suất (score function) luôn bằng 0:

$$\mathbb{E}_{y_i} \left[ \nabla_\theta \log \pi_\theta(y_i | q) \right] = \int \pi_\theta(y_i | q) \frac{\nabla_\theta \pi_\theta(y_i | q)}{\pi_\theta(y_i | q)} dy_i = \nabla_\theta \int \pi_\theta(y_i | q) dy_i = \nabla_\theta (1) = 0$$

Do đó, toàn bộ phần kỳ vọng baseline triệt tiêu và ta thu được:

$$\mathbb{E} \left[ \sum_{i=1}^G \nabla_\theta \log \pi_\theta(y_i | q) A_i \right] = \mathbb{E} \left[ \sum_{i=1}^G \nabla_\theta \log \pi_\theta(y_i | q) R(q, y_i) \right]$$

**Kết luận**: Kỳ vọng của gradient GRPO bằng chính xác kỳ vọng của gradient chính sách gốc. Ước lượng baseline nhóm của GRPO là **hoàn toàn không chệch (unbiased)**.

---

## 2. Công thức lợi thế chuẩn hóa nhóm và giảm phương sai

Thay vì chỉ trừ đi trung bình, GRPO chuẩn hóa điểm thưởng trong nhóm bằng cách chia cho độ lệch chuẩn (standard deviation):

$$A_i = \frac{r_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$

Trong đó $R = \{r_1, r_2, ..., r_G\}$ là tập hợp điểm thưởng của nhóm.

### Tại sao phép chia này giúp giảm phương sai?
1. **Khống chế biên độ (Bounded Gradients)**: Việc chia cho độ lệch chuẩn đưa toàn bộ lợi thế tương đối $A_i$ về phân phối chuẩn có trung bình bằng 0 và độ lệch chuẩn bằng 1. Điều này đảm bảo biên độ của gradient cập nhật luôn ổn định, tránh hiện tượng gradient biến nổ (exploding gradients) khi phân phối điểm thưởng của hệ thống bị dịch chuyển.
2. **Không phụ thuộc vào quy mô điểm thưởng**: Nếu Reward Model đột ngột tăng scale điểm (ví dụ từ thang 10 lên thang 100), toán tử chuẩn hóa nhóm của GRPO sẽ tự động triệt tiêu sự thay đổi quy mô này, giúp quá trình huấn luyện Actor hoàn toàn độc lập với thang điểm tuyệt đối của hàm thưởng.

---

## 3. Mã giả tính toán Advantage trong GRPO

Dưới đây là mã giả tính toán Lợi thế GRPO dựa trên Prompt ID (`uid`), tương thích với hiện thực trong `verl/trainer/ppo/core_algos.py`:

```python
import torch

def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, # Shape: (batch_size, seq_len)
    response_mask: torch.Tensor,       # Shape: (batch_size, seq_len)
    uids: list,                        # Danh sách mã định danh Prompt cho từng dòng trong batch
    norm_adv_by_std_in_grpo: bool = True
) -> (torch.Tensor, torch.Tensor):
    
    # 1. Tính tổng điểm thưởng tích lũy cho mỗi câu trả lời
    # (Vì GRPO thường đánh giá điểm thưởng ở cấp độ cuối câu - outcome reward)
    scores = token_level_rewards.sum(dim=-1) # Shape: (batch_size,)
    
    advantages = torch.zeros_like(scores)
    
    # 2. Gom nhóm các câu trả lời thuộc cùng một Prompt
    group_map = {}
    for idx, uid in enumerate(uids):
        if uid not in group_map:
            group_map[uid] = []
        group_map[uid].append(idx)
        
    # 3. Tính toán lợi thế tương đối trong từng nhóm
    for uid, indices in group_map.items():
        group_indices = torch.tensor(indices, dtype=torch.long)
        group_scores = scores[group_indices]
        
        mean = group_scores.mean()
        std = group_scores.std() if len(indices) > 1 else torch.tensor(0.0)
        
        # Chuẩn hóa lợi thế nhóm
        if norm_adv_by_std_in_grpo and std > 1e-8:
            advantages[group_indices] = (group_scores - mean) / std
        else:
            advantages[group_indices] = group_scores - mean
            
    # 4. Phát tán Advantage ngược lại cấp độ token phục vụ tính loss
    # (Lợi thế của câu trả lời được nhân đều cho các token của response)
    token_advantages = advantages.unsqueeze(-1) * response_mask
    
    # Returns tương đương với advantages + baseline (mean)
    returns = token_advantages
    
    return token_advantages, returns
```

## 💡 Liên hệ thực tế mã nguồn verl
Thuật toán này được hiện thực hóa tại tệp [core_algos.py](file:///Users/admin/TuanDung/repos/verl/verl/trainer/ppo/core_algos.py) trong hàm `compute_grpo_outcome_advantage`. `verl` tối ưu hóa toán tử gom nhóm bằng cách tận dụng các phép toán song song hóa mảng trên PyTorch để xử lý nhanh nhất cụm dữ liệu phân tán được gom về Driver.
