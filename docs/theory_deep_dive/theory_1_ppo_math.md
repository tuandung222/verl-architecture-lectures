---
sidebar_position: 2
sidebar_label: "Bài 1: Toán học của PPO & GAE"
---

# Bài 1: Toán học của PPO (GAE, Clipped Objective & Value Estimator)

Thuật toán **PPO (Proximal Policy Optimization)** là một trong những giải thuật Policy Gradient dựa trên vùng tin cậy (Trust Region) phổ biến nhất hiện nay. Để hiểu sâu sắc cách PPO hoạt động, bài viết này đi vào phân tích gốc rễ toán học của hàm mục tiêu Cắt (Clipped Objective) và bộ ước lượng lợi thế GAE.

---

## 1. Hàm mục tiêu Surrogate Loss và Cơ chế Cắt (Clipped Objective)

Trong học tăng cường chính sách truyền thống (Policy Gradient), chúng ta tối ưu hóa hàm mục tiêu:

$$\nabla_\theta J(\theta) = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t \right]$$

Tuy nhiên, việc cập nhật trực tiếp theo gradient này rất dễ dẫn đến hiện tượng dịch chuyển chính sách quá lớn, làm Actor rơi vào vùng không thể hồi phục (Policy collapse). PPO giải quyết bài toán này bằng cách sử dụng **Surrogate Loss** tỷ lệ xác suất:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

Hàm mục tiêu PPO Clipped Surrogate Objective được định nghĩa như sau:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

### Phân tích ý nghĩa toán học của toán tử $\min$:
Toán tử $\min$ thiết lập một cận dưới (pessimistic bound) cho hàm mục tiêu:

* **Khi $\hat{A}_t > 0$ (Hành động tốt hơn mong đợi)**:
  $r_t(\theta)\hat{A}_t$ tăng lên khi $\pi_\theta(a_t | s_t)$ tăng. Tuy nhiên, cơ chế `clip` sẽ giới hạn tỷ lệ này tối đa ở mức $1+\epsilon$. Do đó, mô hình sẽ không nhận thêm điểm thưởng nếu đẩy xác suất hành động này lên quá cao, ngăn việc cập nhật quá đà.
* **Khi $\hat{A}_t < 0$ (Hành động tệ hơn mong đợi)**:
  $r_t(\theta)\hat{A}_t$ giảm đi khi xác suất hành động giảm. Lệnh `clip` giới hạn tỷ lệ tối thiểu ở mức $1-\epsilon$. Hàm $\min$ sẽ chọn giá trị cực tiểu bị cắt, cho phép gradient tiếp tục đẩy xác suất hành động này xuống thấp để loại bỏ hành vi xấu.

---

## 2. Công thức toán học của GAE (Generalized Advantage Estimation)

Lợi thế $\hat{A}_t$ đại diện cho việc hành động $a_t$ tại trạng thái $s_t$ tốt hơn hay tệ hơn mức trung bình dự đoán. GAE là kỹ thuật giảm phương sai hiệu quả cho PPO bằng cách cân bằng giữa lệch (bias) và phương sai (variance).

Đầu tiên, ta định nghĩa sai số Temporal Difference (TD-error) tại thời điểm $t$ thông qua mạng dự báo Critic $V(s)$:

$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE tính lợi thế $\hat{A}_t^{GAE}$ bằng tổng lũy tích chênh lệch TD-error được khấu hao bởi tham số $\lambda \in [0, 1]$:

$$\hat{A}_t^{GAE}(\gamma, \lambda) = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

### Phân tích hai trường hợp cực đoan:
* **Khi $\lambda = 0$ (Phương sai thấp nhất, Độ lệch cao nhất)**:
  $$\hat{A}_t^{GAE}(\gamma, 0) = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
  Chỉ phụ thuộc vào 1 bước đi tiếp theo. Ước lượng này có phương sai rất thấp vì không phụ thuộc vào chuỗi hành động ngẫu nhiên dài phía sau, nhưng bị chệch lớn vì phụ thuộc hoàn toàn vào độ chính xác dự báo của Critic $V$.
* **Khi $\lambda = 1$ (Phương sai cao nhất, Không chệch)**:
  $$\hat{A}_t^{GAE}(\gamma, 1) = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$$
  Tương đương ước lượng Monte Carlo đầy đủ. Ước lượng này hoàn toàn không bị chệch (unbiased) vì sử dụng phần thưởng thực tế tương lai, nhưng có phương sai cực lớn do sự ngẫu nhiên tích lũy qua tất cả các bước đi tiếp theo.

---

## 3. Mã giả tính toán GAE

Dưới đây là mã giả triển khai thuật toán tính Advantage GAE và Returns tương thích với hàm `compute_gae_advantage_return` trong `verl/trainer/ppo/core_algos.py`:

```python
import torch

def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,  # Shape: (batch_size, seq_len)
    values: torch.Tensor,               # Shape: (batch_size, seq_len)
    response_mask: torch.Tensor,        # Shape: (batch_size, seq_len)
    gamma: float = 1.0,
    lam: float = 0.95
) -> (torch.Tensor, torch.Tensor):
    
    batch_size, seq_len = token_level_rewards.shape
    advantages = torch.zeros_like(token_level_rewards)
    
    # Duyệt ngược từ cuối chuỗi về đầu (Backward accumulation)
    last_gae_lam = 0.0
    for t in reversed(range(seq_len)):
        # Tránh tràn mảng ở bước cuối cùng
        next_value = values[:, t + 1] if t + 1 < seq_len else 0.0
        
        # 1. Tính toán TD-error (delta)
        delta = token_level_rewards[:, t] + gamma * next_value - values[:, t]
        
        # 2. Khấu hao sai số lũy tích bằng lambda
        advantages[:, t] = last_gae_lam = delta + gamma * lam * last_gae_lam
        
    # Áp dụng mặt nạ response mask (chỉ tính lợi thế trên các token của câu trả lời)
    advantages = advantages * response_mask
    
    # 3. Tính Returns dùng để huấn luyện Critic (Value targets)
    returns = advantages + values
    
    return advantages, returns
```

## 💡 Liên hệ thực tế mã nguồn verl
Hàm này được hiện thực hóa tại tệp [core_algos.py](file:///Users/admin/TuanDung/repos/verl/verl/trainer/ppo/core_algos.py) với tên gọi `compute_gae_advantage_return`. Trong mã nguồn thực tế, `verl` tối ưu hóa bằng cách viết dưới dạng vectorized sử dụng tensor toán học để tránh vòng lặp `for` chậm chạp của Python, giúp xử lý hàng triệu token song song trên GPU.
