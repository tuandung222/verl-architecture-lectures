---
sidebar_position: 2
sidebar_label: "Case 1: Tái hiện DeepSeek-R1-Zero (TinyZero)"
---

# Case Study 1: Tái hiện DeepSeek-R1-Zero với GRPO (Dự án TinyZero)

**DeepSeek-R1-Zero** mở ra một kỷ nguyên mới khi chứng minh rằng một mô hình ngôn ngữ có thể tự phát triển khả năng lập luận phức tạp (Reasoning) thông qua Học tăng cường (RL) mà không cần bất kỳ dữ liệu chuỗi suy nghĩ (Chain of Thought - CoT) nào do con người biên soạn. 

Trong Case Study này, chúng ta sẽ khảo sát cách tái hiện lại DeepSeek-R1-Zero dựa trên thuật toán **GRPO** của `verl` để giải quyết các bài toán toán học (tập dữ liệu GSM8K/MATH).

---

## 1. Cơ chế chấm điểm thưởng kép (Dual-Reward System)

Để mô hình vừa suy nghĩ logic vừa tuân thủ định dạng hiển thị, ta triển khai một hệ thống chấm điểm gồm hai thành phần: **Thưởng định dạng (Format Reward)** và **Thưởng nội dung (Accuracy Reward)**.

```
                  ┌──────────────────────────────────────────┐
                  │          Câu trả lời của Actor           │
                  └────────────────────┬─────────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
        [ Kiểm tra định dạng ]                 [ Kiểm tra kết quả ]
     (Phải chứa thẻ <think> và           (Đáp án cuối cùng có khớp với
      <answer> đúng vị trí)                      Ground Truth?)
                     │                                   │
             Đúng: +0.13 điểm                     Đúng: +1.0 điểm
             Sai: 0.0 điểm                        Sai: 0.0 điểm
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
                            [ Tổng điểm thưởng ]
```

### Ví dụ hiện thực hóa hàm thưởng bằng Python:

```python
import re

def accuracy_reward_fn(solution: str, ground_truth: str) -> float:
    """Kiểm tra xem đáp án cuối cùng của mô hình có khớp với Ground Truth hay không"""
    # Trích xuất phần đáp án nằm trong thẻ <answer>...</answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
    if not answer_match:
        return 0.0
    
    extracted_answer = answer_match.group(1).strip()
    # So sánh chuẩn hóa chuỗi hoặc số học
    return 1.0 if extracted_answer == ground_truth.strip() else 0.0

def format_reward_fn(solution: str) -> float:
    """Kiểm tra tính hợp lệ của định dạng suy nghĩ"""
    # Regex kiểm tra cấu trúc: <think> suy nghĩ </think> <answer> đáp án </answer>
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    match = re.match(pattern, solution.strip(), re.DOTALL)
    
    return 0.13 if match else 0.0
```

---

## 2. Cấu hình kịch bản huấn luyện GRPO (`run_tinyzero.sh`)

Chúng ta sẽ cấu hình tệp lệnh chạy để huấn luyện mô hình nền tảng (ví dụ: `Qwen2.5-3B-Instruct`) bằng thuật toán GRPO trong `verl`:

```bash
# run_tinyzero.sh
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.space_max_token_len=2048 \
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    data.train_files=data/gsm8k/train.jsonl \
    data.val_files=data/gsm8k/test.jsonl \
    trainer.total_epochs=3 \
    trainer.logger=['wandb']
```

### Giải thích các tham số cốt lõi:
* `algorithm.adv_estimator=grpo`: Kích hoạt bộ ước lượng GRPO (loại bỏ mô hình Critic).
* `actor_rollout_ref.rollout.n=4`: Group Size $G=4$. Với mỗi prompt đầu vào, hệ thống sẽ sinh ra 4 câu trả lời độc lập để chuẩn hóa điểm thưởng tương đối trong nhóm.
* `actor_rollout_ref.rollout.temperature=1.0`: Đặt nhiệt độ cao để mô hình tích cực khám phá (exploration) các chuỗi suy nghĩ khác nhau, giúp thuật toán tìm ra đường đi tối ưu.
* `actor_rollout_ref.rollout.space_max_token_len=2048`: Tăng giới hạn token sinh ra vì chuỗi suy nghĩ (CoT) sẽ ngày càng dài ra trong quá trình huấn luyện.

---

## 3. Phân tích các xu hướng trên WandB: Hiện tượng "Aha! Moment"

Khi theo dõi tiến trình huấn luyện trên WandB, chúng ta sẽ thấy 3 đồ thị đặc trưng:

1. **`actor/reward_format` (Điểm định dạng)**: Đạt đỉnh tối đa ($0.13$) rất nhanh (chỉ sau khoảng 50-100 steps). Mô hình học được rằng để nhận điểm thưởng, việc đầu tiên là phải viết đúng thẻ `<think>` và `<answer>`.
2. **`actor/reward_accuracy` (Độ chính xác)**: Tăng chậm hơn, đi kèm với sự biến động mạnh do mô hình đang thử nghiệm các hướng lập luận.
3. **`actor/rollout_seq_len` (Độ dài chuỗi sinh mẫu)**: Ban đầu rất ngắn, nhưng sau đó tăng đột ngột (ví dụ từ 300 tokens nhảy lên 1200 tokens).

```
Rollout Sequence Length (Độ dài suy nghĩ tự phát):
Length (Tokens)
  ▲
2000 |                                   /---------
1500 |                                  /
1000 |                    /------------/  <- Aha! Moment (Mô hình phát hiện ra
 500 |    _______________/                       rằng suy nghĩ dài hơn giúp
     └─────────────────────────────────────────────► Steps
```

### Khái niệm "Aha! Moment":
Đây là thời điểm mô hình tự phát hiện ra rằng việc **dành thêm nhiều bước để suy nghĩ, thử nghiệm và tự sửa sai** trong thẻ `<think>` sẽ giúp nó đạt được độ chính xác cao hơn ở thẻ `<answer>`. Độ dài chuỗi nhảy vọt chính là minh chứng vật lý của khả năng tự tư duy mà không cần bất kỳ sự can thiệp hay mớm dữ liệu nào từ con người.
