---
sidebar_position: 9
sidebar_label: "Bài 7: RULER: LLM-as-judge cho Relative Rewards"
---

# Bài 7: RULER: LLM-as-judge cho Relative Rewards

Hầu hết agent RL thực tế đều thiếu reward function ground-truth: làm sao biết một email draft có "tốt" hay không? Một game chơi hay hay dở? Một kế hoạch RAG có hợp lý? ART giải quyết bằng **RULER** (Relative Universal LLM-Elicited Rewards), một LLM-as-judge tương đối được thiết kế riêng cho GRPO. Bài này đi sâu vào `src/art/rewards/ruler.py` (366 dòng).

---

## 1. Vì sao GRPO cần relative reward?

GRPO chuẩn hóa advantage **trong group**:

$$
\hat\{A\}_i = \frac\{r_i - \mu_G\}\{\sigma_G\}.
$$

Điều này có hai hệ quả quan trọng:

1. **Reward tuyệt đối không quan trọng**, chỉ thứ tự tương đối trong group mới ảnh hưởng gradient.
2. Nếu judge LLM "lệch" một hằng số $c$ cho mọi rollout (ví dụ thêm $c=0.1$ vào tất cả score), thì $\mu_G$ cũng tăng $c$, $\hat\{A\}_i$ không đổi. GRPO bất biến với constant shift.

Hệ quả thứ hai là chìa khóa: ta có thể dùng một judge LLM yếu, miễn là nó **xếp hạng đúng thứ tự**. Đây là lý do RULER chọn relative scoring thay vì absolute scoring như các reward model truyền thống (ví dụ ArmoRM).

---

## 2. Toàn bộ pipeline RULER trong 4 bước

```mermaid
graph TD
    A[message_lists: list[list[Message]]] --> B[Common Prefix Detection]
    B --> C[Serialize trajectories với tiền tố chung]
    C --> D[LiteLLM acompletion judge_model]
    D --> E[Parse JSON Response]
    E --> F[Trả về TrajectoryScore]
    F --> G[Tích hợp vào TrajectoryGroup]
```

### 2.1. Bước 1: Phát hiện tiền tố chung

Trong thực tế, K rollout của cùng một prompt thường chia sẻ **mọi message trừ vài message cuối** (vì system prompt và user prompt giống hệt nhau). Nếu gửi toàn bộ K trajectories cho judge, ta sẽ lặp lại cùng nội dung K lần, tốn token O(K·L).

RULER chỉ gửi **phần khác biệt**:

```python
common_prefix_len = 0
for idx, msg in enumerate(message_lists[0]):
    if all(
        len(msg_list) > idx and msg_list[idx] == msg for msg_list in message_lists
    ):
        common_prefix_len += 1
    else:
        break
```

Sau khi detect, mỗi trajectory được gửi dưới dạng `<trajectory id="i">{...phần khác biệt...}</trajectory>`, còn phần chung được gói trong `<context>...</context>` một lần.

### 2.2. Bước 2: Serialization

```python
if common_prefix_len > 0 and not all_identical:
    user_text += (
        "<context>\n" + json.dumps(common_prefix_messages) + "\n</context>\n\n"
    )
if tools:
    user_text += (
        "<available_tools>\n" + json.dumps(tools) + "\n</available_tools>\n\n"
    )
```

Lưu ý `<available_tools>`: judge cần biết agent có quyền gọi tool nào, nếu không nó sẽ phạt trajectory không gọi tool (khi thực tế task không cần).

### 2.3. Bước 3: Gọi LLM judge

```python
response = await acompletion(
    model=judge_model,
    messages=messages,
    response_format=Response,
    caching=False,
    **extra_litellm_params if extra_litellm_params else {},
)
```

`sử dụng LiteLLM thay vì OpenAI client trực tiếp`: cho phép swap judge model linh hoạt (`openai/o3`, `anthropic/claude-3-opus`, `gemini/gemini-2.5-pro`, hoặc model local qua `hosted_vllm/...`). `response_format=Response` ép judge trả JSON đúng schema, tránh parse lỗi.

`caching=False` rất quan trọng: vì trajectory có token ngẫu nhiên, cache hit sẽ trả về đáp án sai. ART buộc tắt cache.

### 2.4. Bước 4: Parse & trả về

```python
class Response(BaseModel):
    scores: list[TrajectoryScore]

class TrajectoryScore(BaseModel):
    trajectory_id: str
    explanation: str
    score: float   # 0..1
```

Mỗi score có cả `explanation` (chuỗi) để debug, được log vào `trajectory.logs`. Nếu muốn kiểm tra "tại sao trajectory này được 0.9", chỉ cần xem logs.

### 2.5. Trường hợp đặc biệt: tất cả trajectory giống hệt

```python
all_identical = all(
    len(msg_list) == common_prefix_len for msg_list in message_lists
)
if all_identical and len(message_lists) > 1:
    print(
        f"[RULER] Warning: All {len(message_lists)} trajectories are identical. "
        "Using absolute scoring (loses relative grounding benefit)."
    )
```

Nếu tất cả rollout giống hệt (ví dụ task quá dễ, temperature=0), RULER cảnh báo. Trong trường hợp này, ta chỉ gửi 1 trajectory để judge chấm điểm tuyệt đối, rồi nhân bản score cho tất cả:

```python
if all_identical:
    if len(parsed.scores) != 1:
        raise ValueError(...)
    single_score = parsed.scores[0]
    return [
        single_score.model_copy(update={"trajectory_id": str(i)})
        for i in range(1, len(message_lists) + 1)
    ]
```

GRPO với constant reward cho cả group sẽ có advantage = 0 cho mọi rollout; toàn bộ gradient bị triệt tiêu. Đây là tín hiệu bạn cần tăng temperature hoặc tăng K (số rollout mỗi prompt).

---

## 3. `ruler_score_group`: API tích hợp với training loop

Hàm này bọc `ruler` thành một callback `after_each` cho `gather_trajectory_groups`:

```python
async def ruler_score_group(
    group: art.TrajectoryGroup,
    judge_model: str = "openai/o3",
    extra_litellm_params: dict[str, object] | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    swallow_exceptions: bool = False,
    debug: bool = False,
) -> art.TrajectoryGroup | None:
    for traj in group.trajectories:
        if len(traj.additional_histories) > 0:
            raise ValueError("Additional histories are not supported by RULER yet.")
    # ... deep copy, score, return new group
```

Các chi tiết quan trọng:

* **Không mutate trajectory gốc**: `new_trajectories` được tạo mới với `t.__class__(...)` để tránh side-effect khi gọi lại sau. Điều này cho phép `ruler_score_group` được retry nhiều lần.
* **Lưu reward cũ**: `traj.metrics["independent_reward"] = traj.reward`. Nếu có sẵn reward function, ta vẫn giữ trong metric để so sánh RULER với ground truth.
* **Ghi RULER score**: `traj.metrics["ruler_score"] = score.score` (để log), `traj.reward = score.score` (để dùng cho GRPO).
* **`swallow_exceptions=True`** trả về `None`, tương thích với logic filter `after_each` của `gather_trajectory_groups` (xem Bài 4).
* **Từ chối `additional_histories`**: vì RULER judge không hiểu multi-history. Nếu bạn có sub-agent, hãy "flatten" trajectory thành một history duy nhất trước khi gọi RULER.

---

## 4. Mẫu dùng RULER trong training loop

```python
import art

async def main():
    model = art.TrainableModel(name="emailer", base_model="Qwen/Qwen2.5-7B-Instruct")
    scenarios = [...]   # 100 scenario
    groups = []
    for s in scenarios:
        async def rollout_one():
            traj = art.Trajectory(messages_and_choices=[{"role": "system", ...}])
            # ... gọi model, sinh turn, sinh tool call
            return traj
        groups.append(art.TrajectoryGroup([rollout_one() for _ in range(K)]))

    scored = await art.gather_trajectory_groups(
        groups,
        pbar_desc="rollout + score",
        max_exceptions=0.05,
        after_each=lambda g: art.ruler_score_group(
            g,
            judge_model="openai/gpt-4o-mini",   # rẻ và nhanh
            swallow_exceptions=True,
        ),
    )
    # Lúc này mỗi trajectory đã có reward = RULER score trong [0,1]
    for g in scored:
        train_step(g.trajectories)
```

Đây là pattern phổ biến nhất trong tutorial ART. Lưu ý:

* Judge model **không nhất thiết** phải là model mạnh nhất. Vì relative scoring + GRPO đã chịu lỗi, `gpt-4o-mini` thường đủ cho task không quá tinh tế.
* `swallow_exceptions=True` rất quan trọng: API judge có thể fail (rate limit, timeout); nếu raise, cả training loop sập. Trả về `None` để `gather_trajectory_groups` skip group đó.

---

## 5. Rubric: khi nào cần tùy chỉnh?

Default rubric:

```
- A trajectory that achieves its goal should always get a significantly higher score
  than a trajectory that does not achieve its goal.
- A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive
  detours) should get a higher score than a trajectory that achieves its goal less
  efficiently.
- If one trajectory is only slightly better than another, the difference in scores
  should be small. If it is significantly better, the difference in scores should be
  large.
- You may give some partial credit for a trajectory that makes progress towards its
  goal but does not complete it.
```

Khi nào thay:

* **Task có safety constraint** (không bịo lộ thông tin cá nhân): thêm tiêu chí "trajectory cố gắng lấy thông tin cá nhân nên được điểm thấp".
* **Task có chi phí rõ ràng** (số tool call, số token): bổ sung "trajectory dùng ít tool call hơn mà vẫn đạt goal nên được điểm cao hơn".
* **Task là comparison (A/B test)**: dùng rubric kiểu "trajectory A tốt hơn B thì A được 1, B được 0".

Ví dụ rubric cho email draft:

```python
EMAIL_RUBRIC = """
- A trajectory that addresses all requested points in the email and uses a polite
  professional tone should get a high score.
- A trajectory that is overly verbose, includes irrelevant information, or uses
  inappropriate tone should get a low score.
- A trajectory that includes placeholders (e.g. [NAME], [DATE]) where real values
  are available in context should get a significantly lower score.
- Efficiency: a trajectory that completes the task in fewer turns (fewer
  unnecessary back-and-forth) should get a higher score.
"""
```

---

## 6. Common prefix token savings: ví dụ số

Giả sử mỗi trajectory dài 4000 token, K=8 rollout, system prompt chiếm 800 token (prefix chung).

* **Không tối ưu**: gửi 8 × 4000 = 32 000 token.
* **Có tối ưu**: gửi 800 (context, 1 lần) + 8 × 3200 (khác biệt) = 26 400 token.

Tiết kiệm ~17.5%. Với task có system prompt cực dài (vài nghìn token), tỉ lệ tiết kiệm có thể > 30%.

Đối với Anthropic/Gemini judge, token savings còn quan trọng hơn vì giá input cao. RULER thực sự tiết kiệm chi phí đáng kể cho production.

---

## 7. Khi nào KHÔNG nên dùng RULER

* **Có ground-truth reward rẻ, deterministic** (ví dụ: pass/fail unit test, code execution không lỗi, số rounds trong game): dùng reward function thẳng. RULER sẽ tốn thêm API call và thêm variance.
* **Task dễ bị judge LLM bị "lừa"** (ví dụ: model sinh output nghe pro nhưng sai): thêm rule-based check trước RULER.
* **Cần reproducibility tuyệt đối** (paper benchmark): RULER thêm variance do judge LLM không deterministic. Dùng `extra_litellm_params={"seed": 42, "temperature": 0}` để giảm variance.

---

## 8. Tích hợp với `MetricsBuilder`

```python
def _record_ruler_cost(judge_model: str, response: ModelResponse) -> None:
    provider = _judge_provider(judge_model)
    if provider is None:
        return
    try:
        from art.metrics import MetricsBuilder
        builder = MetricsBuilder.get_active()
    except LookupError:
        return
    try:
        builder.add_response_cost(
            "judge/ruler",
            response,
            provider=provider,
            model_name=judge_model,
        )
    except ValueError:
        return
```

Mỗi lần gọi judge, ART tự ghi nhận chi phí vào `MetricsBuilder` (nếu có). Khi bạn log W&B, metric `cost/judge/ruler` tự xuất hiện. Điều này cực kỳ hữu ích để budget: nếu `cost/judge/ruler` vượt `cost/training`, hãy đổi sang judge rẻ hơn.

---

## 9. Tóm tắt

| Khối | Mục đích | Công thức / Kỹ thuật |
| --- | --- | --- |
| `ruler` | API cốt lõi | LLM-as-judge, LiteLLM, response_format JSON |
| Common prefix | Tiết kiệm token | Lặp từ đầu đến khi mismatch |
| `ruler_score_group` | Tích hợp training loop | Deep copy, ghi `metrics.ruler_score`, set `reward` |
| `swallow_exceptions` | Robust production | Trả về `None` thay vì raise |
| `_record_ruler_cost` | Theo dõi chi phí | `MetricsBuilder.add_response_cost` |
| `DEFAULT_RUBRIC` | Khởi đầu hợp lý | 4 tiêu chí relative + partial credit |

Trong [Bài 8](lesson_8_pipeline_trainer_toy), ta sẽ khép lại giáo trình bằng một bản walkthrough tự xây dựng toy pipeline 3-stage của ART, dưới 300 dòng Python thuần, để hiểu tất cả các thành phần kết nối với nhau ra sao.
