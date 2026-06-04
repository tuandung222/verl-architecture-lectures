---
sidebar_position: 5
sidebar_label: "Bài 4: Lý thuyết về KL Divergence & Adaptive Control"
---

# Bài 4: Lý thuyết về KL Divergence & Adaptive Control

Trong quá trình huấn luyện RLHF, mô hình Actor ($\pi_\theta$) liên tục thay đổi trọng số để tối đa hóa điểm thưởng. Tuy nhiên, nếu không có cơ chế kiểm soát, mô hình sẽ nhanh chóng bị "lệch hướng" (policy drift) và tìm ra các kẽ hở của Reward Model (Reward Hacking). 

**KL Divergence (Độ chênh lệch Kullback-Leibler)** là công cụ toán học tối quan trọng dùng để thiết lập vùng tin cậy (trust region), giữ cho Actor không đi quá xa khỏi mô hình gốc Reference ($\pi_{ref}$).

---

## 1. Công thức toán học của KL Divergence trong RLHF

Độ lệch KL giữa hai phân phối xác suất rời rạc $P$ và $Q$ được định nghĩa như sau:

$$D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

Trong RLHF, chúng ta áp dụng hình phạt KL cấp độ token (Token-level KL penalty) cho mỗi token $a_t$ được sinh ra tại trạng thái $s_t$:

$$D_{KL}\left(\pi_\theta(a_t | s_t) || \pi_{ref}(a_t | s_t)\right) = \log \frac{\pi_\theta(a_t | s_t)}{\pi_{ref}(a_t | s_t)}$$

### Biến thể ước lượng của Schulman (Schulman's Approximation):
Trên thực tế, việc sử dụng tỷ lệ log thô $\log(P/Q)$ có phương sai rất lớn và đôi khi cho ra giá trị âm (không hợp lệ với tính chất không âm của KL divergence). John Schulman (tác giả PPO) đã đề xuất một công thức ước lượng xấp xỉ tối ưu hơn:

$$D_{KL}^{\text{approx}} = \frac{Q(x)}{P(x)} - \log \frac{Q(x)}{P(x)} - 1$$

Công thức này sở hữu hai ưu điểm toán học vượt trội:
1. **Luôn không âm**: Hàm $f(x) = x - \log x - 1 \ge 0$ với mọi $x > 0$. Do đó, ước lượng này luôn đảm bảo tính chất không âm của KL divergence ở mọi bước tính toán.
2. **Phương sai cực thấp**: Tránh được việc tính trực tiếp log xác suất của Actor vốn biến động mạnh ở các bước đầu tối ưu hóa.

---

## 2. Cơ chế điều khiển hệ số KL thích ứng (Adaptive KL Controller)

Hàm thưởng tích hợp KL Penalty được định nghĩa như sau:

$$R_{\text{final}} = R_{\text{RM}} - \beta D_{KL}(\pi_\theta || \pi_{ref})$$

Trong đó hệ số $\beta$ điều khiển mức độ phạt. Nếu cố định $\beta$, quá trình huấn luyện rất khó hội tụ:
* Nếu $\beta$ quá lớn: Mô hình bị bó cứng, không thể học được gì mới từ Reward.
* Nếu $\beta$ quá nhỏ: Mô hình nhanh chóng bị Reward Hacking.

Để cân bằng, `verl` triển khai bộ điều khiển **Adaptive KL Controller** (Bộ điều khiển tỷ lệ thích ứng):

* **Mục tiêu**: Giữ cho giá trị trung bình KL Divergence thực tế $d_{\text{current}}$ tiệm cận với giá trị KL mục tiêu $d_{\text{target}}$ (thường đặt $\sim 5.0$ đến $10.0$).
* **Công thức cập nhật hệ số $\beta$**:
  Sau mỗi bước huấn luyện (Global step), hệ thống đo đạc $d_{\text{current}}$ và hiệu chỉnh $\beta$:
  $$\beta \leftarrow \max\left(\beta_{\text{min}}, \beta \cdot (1 + K_p (d_{\text{current}} - d_{\text{target}}))\right)$$
  Trong đó $K_p$ là hệ số tỷ lệ (Proportional gain, thường chọn $0.1$).
  * Nếu $d_{\text{current}} > d_{\text{target}}$ (Actor đang dịch chuyển quá nhanh): Tăng hệ số phạt $\beta$ để kéo Actor lại gần Reference.
  * Nếu $d_{\text{current}} < d_{\text{target}}$ (Actor đang quá thụ động): Giảm $\beta$ để giải phóng không gian khám phá tự do cho Actor.

---

## 3. Mã giả tính toán KL Divergence và Cập nhật Hệ số $\beta$

Dưới đây là mã giả thể hiện cách tính KL penalty cấp token và thuật toán tự hiệu chỉnh hệ số $\beta$ trên Driver:

```python
import torch
import math

class AdaptiveKLController:
    def __init__(self, init_beta: float, target_kl: float, kp: float = 0.1):
        self.beta = init_beta
        self.target_kl = target_kl
        self.kp = kp
        self.min_beta = 1e-6

    def update(self, current_kl: float, n_steps: int):
        """Cập nhật hệ số beta dựa trên sai số KL"""
        # Công thức hiệu chỉnh sai số tỷ lệ
        error = (current_kl - self.target_kl) / self.target_kl
        self.beta = self.beta * (1.0 + self.kp * error)
        self.beta = max(self.min_beta, self.beta)

def compute_kl_penalty(
    actor_log_probs: torch.Tensor,  # Shape: (batch_size, seq_len)
    ref_log_probs: torch.Tensor,    # Shape: (batch_size, seq_len)
    response_mask: torch.Tensor,    # Shape: (batch_size, seq_len)
    kl_penalty_type: str = "kl"
) -> torch.Tensor:
    
    # Tỷ lệ log: log(P) - log(Q)
    log_ratio = actor_log_probs - ref_log_probs
    
    if kl_penalty_type == "kl":
        # KL tiêu chuẩn
        kl = log_ratio
    elif kl_penalty_type == "approx_schulman":
        # Công thức ước lượng xấp xỉ của Schulman
        ratio = torch.exp(-log_ratio) # Q / P
        kl = ratio - log_ratio - 1.0
    else:
        raise ValueError(f"Unknown KL type: {kl_penalty_type}")
        
    # Áp dụng mặt nạ chỉ tính KL trên response tokens
    kl = kl * response_mask
    return kl
```

## 💡 Liên hệ thực tế mã nguồn verl
Hàm tính toán KL và bộ điều khiển thích ứng nằm tại tệp [core_algos.py](file:///Users/admin/TuanDung/repos/verl/verl/trainer/ppo/core_algos.py) (`kl_penalty`, `AdaptiveKLController`) và được Driver gọi tích hợp tại phương thức `apply_kl_penalty` trong file [ray_trainer.py](file:///Users/admin/TuanDung/repos/verl/verl/trainer/ppo/ray_trainer.py). Nhờ có cơ chế này, quá trình huấn luyện RLHF trong `verl` diễn ra vô cùng ổn định và tránh được các lỗi suy sụp chính sách thường gặp ở các framework cũ.
