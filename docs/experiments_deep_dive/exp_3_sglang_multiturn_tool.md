# Bài 3: Huấn luyện Multi-turn Agent và Tool Calling tích hợp SGLang

## 1. Bài toán thực tế (Use Case)

Các mô hình ngôn ngữ lớn hiện nay không chỉ dừng lại ở việc tạo văn bản đơn thuần, mà cần phải tương tác với môi trường bên ngoài dưới vai trò là các Tác tử (Agents). Ví dụ:
* Tự động chạy mã kiểm tra trong một **Code Sandbox** (Code Interpreter).
* Truy vấn thông tin thời gian thực qua **Web Search API** để bổ sung kiến thức.
* Thực hiện tính toán phức tạp thông qua các công cụ máy tính (Calculators).

Các tác vụ này yêu cầu mô hình thực hiện tương tác đa lượt (Multi-turn): Suy nghĩ $\rightarrow$ Gọi công cụ $\rightarrow$ Nhận phản hồi từ môi trường $\rightarrow$ Tiếp tục suy nghĩ và đưa ra câu trả lời cuối cùng. Quá trình này được gọi là chu trình **ReAct (Reasoning + Acting)**.

---

## 2. Quy trình tương tác đa lượt (Multi-turn Rollout) với SGLang

Trong `verl`, pha Rollout đa lượt được quản lý thông qua sự phối hợp với công cụ thực thi **SGLang**. 

Khi Actor (mô hình) sinh mẫu, SGLang đóng vai trò là một trình thông dịch hội thoại động:
1. **Phát hiện cuộc gọi công cụ (Tool Call Detection)**: SGLang quét chuỗi token được sinh ra. Nếu xuất hiện một thẻ đặc biệt (ví dụ: `<tool_call>` hoặc chuỗi JSON định dạng OpenAI Tool Call), SGLang sẽ lập tức chặn luồng sinh (yield).
2. **Thực thi công cụ (Tool Execution)**: Trình quản lý của `verl` chuyển đổi tham số trong thẻ gọi công cụ và thực thi mã tương ứng (ví dụ: chạy code Python trong một sandbox docker, hoặc gửi HTTP request đến MCP Server).
3. **Chèn kết quả (Observation Injection)**: Kết quả trả về của công cụ được định dạng thành một lượt thoại mới của hệ thống (ví dụ: `<observation>...</observation>`) và nối vào cuối context window của mô hình.
4. **Tiếp tục sinh (Resume Generation)**: SGLang kích hoạt lại mô hình để tiếp tục sinh chuỗi phản hồi cho lượt tiếp theo.

---

## 3. Che mặt nạ mất mát (Loss Masking) trong huấn luyện Multi-turn

Một lỗi nghiêm trọng thường gặp khi tự triển khai huấn luyện RL cho Agent đa lượt là để mô hình học (tính gradient) trên toàn bộ chuỗi hội thoại. Điều này khiến Actor tìm cách tối ưu hóa (học thuộc) cả các phản hồi từ môi trường hoặc từ công cụ ngoài (Observations), vốn là các dữ liệu do hệ thống sinh ra chứ không phải do mô hình tạo ra.

Để giải quyết, `verl` áp dụng cơ chế **Loss Masking** cực kỳ nghiêm ngặt:

```text
Prompt (Câu hỏi)        -> loss_mask = 0  (Không tính gradient)
Thought (Mô hình nghĩ)  -> loss_mask = 1  (Tính gradient để cập nhật chính sách)
Tool Call (Gọi công cụ) -> loss_mask = 1  (Tính gradient để học cách gọi công cụ)
Observation (Kết quả)   -> loss_mask = 0  (Không tính gradient)
Final Answer (Đáp án)   -> loss_mask = 1  (Tính gradient)
```

Chỉ các token do chính mô hình sản sinh (ở trạng thái `loss_mask = 1`) mới được đưa vào tính toán hàm mất mát Policy Gradient (Surrogate Loss). Các token còn lại được xem như là điều kiện biên (context) của môi trường.

---

## 4. Hàm thưởng định hình (Toolcall Shaping Reward)

Trong giai đoạn đầu của quá trình huấn luyện RL, chính sách của mô hình còn rất ngẫu nhiên (high entropy). Xác suất để mô hình tự sinh ra chính xác cú pháp gọi tool (ví dụ: viết đúng định dạng XML `<tool_call>{"name": "calculator", "args": {"expr": "2+2"}}</tool_call>`) là cực kỳ thấp. Nếu không gọi được công cụ, mô hình sẽ không nhận được bất kỳ điểm thưởng hoàn thành nhiệm vụ nào, dẫn đến việc quá trình huấn luyện RL bị tắc nghẽn hoàn toàn.

Để khắc phục hiện tượng này, chúng ta áp dụng **Hành vi định hình phần thưởng (Reward Shaping)**.

Hãy phân tích mã nguồn định nghĩa hàm thưởng tại:
[gsm8k_toolcall_shaping.py](file:///Users/admin/TuanDung/repos/verl/examples/sglang_multiturn/gsm8k_toolcall_shaping/gsm8k_toolcall_shaping.py)

```python
def toolcall_shaping_reward(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    *,
    shaping_reward: float = 0.1,
    trigger_substring: str = "<tool_call>",
    **kwargs,
) -> float:
    # 1. Tính toán điểm số chính xác của bài toán (GSM8K Base Score: 0.0 hoặc 1.0)
    base = gsm8k_compute_score(solution_str, ground_truth)

    # 2. Điểm thưởng định hình (Shaping Bonus): Cộng điểm nếu mô hình kích hoạt thành công tool
    bonus = shaping_reward if (trigger_substring and trigger_substring in solution_str) else 0.0
    
    return float(base + bonus)
```

Nhờ có điểm thưởng `bonus = 0.1` này, ngay cả khi mô hình chưa giải đúng toán nhưng đã biết viết ra thẻ gọi tool `<tool_call>`, nó vẫn nhận được phản hồi tích cực từ môi trường để tiếp tục tối ưu hóa hành vi này trong các epoch tiếp theo.

---

## 5. Phân tích Kịch bản Huấn luyện thực tế

Kịch bản huấn luyện GRPO kết hợp gọi tool được khai báo tại:
[run_gsm8k_grpo_toolcall_shaping.sh](file:///Users/admin/TuanDung/repos/verl/examples/sglang_multiturn/gsm8k_toolcall_shaping/run_gsm8k_grpo_toolcall_shaping.sh)

Các tham số hệ thống then chốt bao gồm:
* `actor_rollout_ref.rollout.name=sglang`: Sử dụng SGLang làm công cụ sinh mẫu để quản lý vòng lặp hội thoại đa lượt tương tác.
* `reward.custom_reward_function.path`: Chỉ định đường dẫn tới file tính điểm thưởng định hình `gsm8k_toolcall_shaping.py`.
* `actor_rollout_ref.rollout.multi_turn.tool_config_path`: Chỉ định đường dẫn tới cấu hình YAML của các công cụ (ví dụ chứa định nghĩa của `Gsm8kTool` và tham số mô tả của hàm `calc_gsm8k_reward` để SGLang nạp vào hệ thống).
