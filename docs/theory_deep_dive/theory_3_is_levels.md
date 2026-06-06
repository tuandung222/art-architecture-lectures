---
sidebar_position: 4
sidebar_label: "Theory 3: Importance Sampling Levels"
---

# Theory 3: Importance Sampling Levels: 4 cách aggregate ratio

Bài này phân tích 4 cách ART aggregate importance-sampling ratio $\rho$ từ **token-level** lên **sequence-level** và các biến thể, kèm phân tích bias-variance trade-off, và code trong `loss.py`.

---

## 1. Tại sao cần aggregate ratio?

Ở PPO/CISPO cổ điển, $\rho_t = \pi_\theta(a_t|s_t) / \pi_\{\text\{old\}\}(a_t|s_t)$ cho **mỗi token**. Với rollout ngắn (50 token), variance của $\rho$ thấp. Nhưng với rollout dài (500-2000 token):

* Một số token có $\pi_\{\text\{old\}\} \approx 0$ (model không hề "nghĩ" đến action này trước đó) -> $\rho$ cực lớn.
* Một số token có $\pi_\theta$ tăng đột biến do exploration.
* Variance $\rho$ cao -> gradient noisy.

Sequence-level ratio là một cách **giảm variance** bằng cách gộp qua cả rollout. Ý tưởng: nếu rollout tổng thể là "tốt" (high reward) hoặc "xấu" (low reward), ta không cần IS weight chi tiết từng token; một số vô hướng đủ.

---

## 2. Bốn cấp trong ART

Trong `src/art/loss.py`:

```python
importance_sampling_level = experimental_config.get("importance_sampling_level", "token")
prob_ratio = torch.exp(logprob_diff)
if importance_sampling_level != "token":
    sequence_prob_ratio = torch.exp(
        aligned_inputs.group_mean(
            logprob_diff,
            by=aligned_inputs.group_ids * assistant_mask,
        )
    )
    if importance_sampling_level == "sequence":
        prob_ratio = sequence_prob_ratio
    elif importance_sampling_level == "average":
        prob_ratio = (prob_ratio + sequence_prob_ratio) / 2
    elif importance_sampling_level == "geometric_average":
        prob_ratio = (prob_ratio**0.5) * (sequence_prob_ratio**0.5)
```

### 2.1. `token` (mặc định)

$$
\rho_t = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_\{\text\{old\}\}(a_t|s_t))
$$

Mỗi token có weight riêng. Variance cao, bias thấp. Phù hợp rollout ngắn.

### 2.2. `sequence`

$$
\rho_\{\text\{seq\}\} = \exp\left( \frac\{1\}\{|T|\} \sum_\{t \in T\} (\log \pi_\theta(a_t|s_t) - \log \pi_\{\text\{old\}\}(a_t|s_t)) \right)
$$

Một số vô hướng cho cả rollout. Variance thấp hơn nhiều, nhưng mất tín hiệu per-token. Phù hợp rollout dài và reward ở cuối.

### 2.3. `average`

$$
\rho_t^\{\text\{avg\}\} = \frac\{1\}\{2\}(\rho_t + \rho_\{\text\{seq\}\})
$$

Trung bình cộng. Kết hợp signal per-token và global. Phù hợp rollout trung bình (200-500 token).

### 2.4. `geometric_average`

$$
\rho_t^\{\text\{geo\}\} = \sqrt\{\rho_t \cdot \rho_\{\text\{seq\}\}\} = \exp\left( \frac\{1\}\{2\}(\log \rho_t + \log \rho_\{\text\{seq\}\}) \right)
$$

Trung bình nhân (trong log space). Tương tự `average` nhưng "fair" hơn khi scale lớn. Phù hợp rollout có mix token dễ và khó.

---

## 3. `group_mean` helper

`aligned_inputs.group_mean(logprob_diff, by=...)` được implement trong `AlignedLossInputs`:

```python
def group_mean(self, values: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
    return group_aggregate(values, by=by, reduce="mean")
```

Trong `src/art/utils/group_aggregate.py`:

```python
def group_aggregate(
    values: torch.Tensor,
    *,
    by: torch.Tensor,
    reduce: Reduction = "mean",
    eps: float = 1e-8,
    broadcast: bool = True,
) -> torch.Tensor:
    """Vectorised group-wise reduction over sequence dimension."""
```

Hàm này tính mean (hoặc sum/var/std) theo nhóm. `by=aligned_inputs.group_ids * assistant_mask` đảm bảo chỉ average trên token assistant (không tính token user/system).

Một điểm tinh tế: `group_ids * assistant_mask` -> token assistant giữ group_id, token khác có id=0. Kết quả: chỉ token assistant trong cùng group được average. Đây là một "masking trick" khá gọn.

---

## 4. Phân tích variance

Giả sử rollout có T token. Coi log ratio $\delta_t = \log \pi_\theta - \log \pi_\{\text\{old\}\}$ là biến ngẫu nhiên iid với mean $\mu$ và std $\sigma$.

### 4.1. `token` ratio

$$
\rho_t = \exp(\delta_t)
$$

$$
\mathbb\{E\}[\rho_t] = \exp(\mu + \sigma^2/2) \quad \text\{(lognormal mean)\}
$$
$$
\text\{Var\}[\rho_t] = (\exp(\sigma^2) - 1) \exp(2\mu + \sigma^2)
$$

Variance tăng theo cấp số nhân với $\sigma^2$. Với $\sigma = 0.5$ (realistic), variance có thể rất lớn.

### 4.2. `sequence` ratio

$$
\rho_\{\text\{seq\}\} = \exp\left( \frac\{1\}\{T\} \sum_t \delta_t \right)
$$

Với T lớn, $\frac\{1\}\{T\} \sum \delta_t \sim \mathcal\{N\}(\mu, \sigma^2/T)$. Variance của $\rho_\{\text\{seq\}\}$ tỉ lệ nghịch với T. **Variance giảm sqrt(T) lần so với token ratio**.

### 4.3. So sánh

| Level | Var reduction | Bias | Phù hợp |
| --- | --- | --- | --- |
| `token` | 1x (baseline) | 0 | T < 100 |
| `sequence` | T lần | Hơi bias nếu rollout dài không đều | T > 500 |
| `average` | ~ T/4 lần | Trung bình | 100 < T < 500 |
| `geometric_average` | ~ T/4 lần | Trung bình | 100 < T < 500 |

Khi rollout rất dài (2000+ token) và reward ở cuối (không thể biết token nào "quan trọng"), `sequence` thắng rõ rệt.

---

## 5. Bias analysis

Vì $\rho_\{\text\{seq\}\}$ dùng mean của log ratio, nó **không phải unbiased estimator** của mean $\rho_t$. Thực tế:

$$
\mathbb\{E\}[\rho_\{\text\{seq\}\}] = \exp(\mu) \cdot \mathbb\{E\}\left[\exp\left( \frac\{1\}\{T\} \sum (\delta_t - \mu) \right)\right] = \exp(\mu) \cdot \exp(\sigma^2 / (2T))
$$

Nghĩa là $\rho_\{\text\{seq\}\}$ hơi inflated. Với T lớn, bias gần 0.

Với rollout có token "đặc biệt" (token quan trọng, quyết định success/fail), `sequence` ratio **không phân biệt được** token đó. Đây là lý do `average` và `geometric_average` tồn tại: giữ tín hiệu per-token nhưng giảm variance.

---

## 6. Khi nào chọn level nào

### 6.1. Token-level (`token`)

* Rollout ngắn (< 100 token).
* Cần tín hiệu chính xác per-token.
* Reward dense (mỗi token có feedback).

Ví dụ: text generation thuần, code completion.

### 6.2. Sequence-level (`sequence`)

* Rollout dài (> 500 token).
* Reward ở cuối episode.
* Variance per-token cao (model mới, đang học).

Ví dụ: 2048 game (rollout 50-200 turn), MCP tool use.

### 6.3. Average (`average`)

* Rollout trung bình (100-500 token).
* Muốn balance.

Ví dụ: email draft 200-500 token.

### 6.4. Geometric (`geometric_average`)

* Tương tự `average` nhưng khi scale ratio rất khác nhau.

---

## 7. Tích hợp với `prob_ratio.detach()`

Trong nhánh `if not ppo:` (CISPO), `prob_ratio` được detach:

```python
prob_ratio = torch.exp(logprob_diff)
if importance_sampling_level != "token":
    sequence_prob_ratio = torch.exp(...)
    if importance_sampling_level == "sequence":
        prob_ratio = sequence_prob_ratio
    elif importance_sampling_level == "average":
        prob_ratio = (prob_ratio + sequence_prob_ratio) / 2
    elif importance_sampling_level == "geometric_average":
        prob_ratio = (prob_ratio**0.5) * (sequence_prob_ratio**0.5)
# ...
policy_loss = -(
    torch.clip(prob_ratio.detach(), 1 - epsilon, 1 + epsilon_high)
    * advantages
    * new_logprobs
)
```

Lưu ý: `prob_ratio.detach()` chỉ áp dụng cuối cùng. Trong nhánh `average` và `geometric_average`, ta **không detach ngay** vì vẫn cần tính `sequence_prob_ratio` (có gradient vì phụ thuộc `new_logprobs`).

Câu hỏi: gradient có chảy qua `sequence_prob_ratio` không? Câu trả lời: **không**, vì ở dòng cuối `prob_ratio.detach()`. Nhưng trong trung gian, autograd vẫn track. Hiệu quả cuối: gradient chỉ chảy qua `new_logprobs` (term thứ 3 trong loss), không qua `prob_ratio`.

---

## 8. Kết hợp với TIS

Nếu train multi-step trên cùng batch (Bài 5 main), `prob_ratio` còn được điều chỉnh bởi **TIS** (truncated importance sampling):

```python
if upper_bound := experimental_config.get("truncated_importance_sampling", None):
    if aligned_inputs.original_logprobs is not None:
        ...
        logprob_diff = old_logprobs - original_logprobs
        prob_ratio = torch.exp(logprob_diff)
    policy_loss *= torch.clamp(prob_ratio, max=upper_bound).detach()
```

Khi này `prob_ratio` (sau khi gán lại ở TIS) là **TIS weight** (giữa `old` và `original`). `policy_loss` được scale thêm bởi `min(TIS_weight, upper_bound)`. Multi-step càng nhiều thì TIS càng quan trọng.

---

## 9. Kết hợp với masking

Cuối cùng:

```python
policy_loss = policy_loss * weights * assistant_mask
denominator = aligned_inputs.denominator(assistant_mask, reduction)
reduced_policy_loss = policy_loss.sum() / denominator
```

`assistant_mask` chỉ giữ token do assistant sinh. Token user/system/tool được mask = 0, không đóng góp gradient.

`weights` cho phép up-weight/down-weight rollout hoặc token. Ví dụ: trong multi-task learning, weight cao cho task quan trọng.

---

## 10. Kết nối với KL penalty

`kl_penalty_coef > 0` thêm penalty vào `advantages`:

```python
if kl_penalty_coef > 0 and ref_logprobs is not None:
    kl_per_token = (new_logprobs - ref_logprobs).detach() * assistant_mask
    avg_kl = aligned_inputs.masked_mean(kl_per_token, assistant_mask)
    kl_penalty = kl_penalty_coef * (avg_kl - kl_per_token) * assistant_mask
    advantages = advantages + kl_penalty
```

Đây là "thêm vào advantage" thay vì "thêm vào loss". Ưu: KL penalty có ý nghĩa rõ ràng (advantage âm cho token có KL lớn). Nhược: detaching `kl_per_token` có thể gây gradient sai khi combine với IS weight. Chi tiết ở [Theory 4: KL Penalty](theory_4_kl_penalty).

---

## 11. Pseudocode cho cả 4 level

```python
def compute_prob_ratio(new_logp, old_logp, group_ids, assistant_mask, level):
    """Trả về prob_ratio tensor theo level."""
    logprob_diff = new_logp - old_logp
    if level == "token":
        return torch.exp(logprob_diff)
    # Các level khác cần group_mean
    sequence_log_ratio = group_mean(
        logprob_diff, by=group_ids * assistant_mask
    )
    sequence_prob_ratio = torch.exp(sequence_log_ratio)
    if level == "sequence":
        return sequence_prob_ratio
    token_prob_ratio = torch.exp(logprob_diff)
    if level == "average":
        return (token_prob_ratio + sequence_prob_ratio) / 2
    if level == "geometric_average":
        return (token_prob_ratio.sqrt() * sequence_prob_ratio.sqrt())
    raise ValueError(f"Unknown level: {level}")
```

Trong thực tế, `loss_fn` thực hiện inline để tránh overhead function call.

---

## 12. Ví dụ số

Rollout 100 token, tất cả token có `log_ratio = 0.1` (model tăng nhẹ confidence). group_id giống nhau.

* `token`: $\rho_t = e^\{0.1\} \approx 1.105$ cho mỗi token.
* `sequence`: $\rho_\{\text\{seq\}\} = e^\{0.1\} \approx 1.105$ (vì mean = 0.1).
* `average`: $(1.105 + 1.105) / 2 = 1.105$.
* `geometric`: $\sqrt\{1.105 \times 1.105\} = 1.105$.

Trong trường hợp này cả 4 level giống nhau (vì log ratio đều).

Rollout 100 token, một nửa log_ratio = -0.2, nửa kia = +0.4. Mean = 0.1 (giống ví dụ trên).

* `token`: một nửa $\rho = e^\{-0.2\} \approx 0.819$, nửa kia $\rho = e^\{0.4\} \approx 1.492$. Variance cao.
* `sequence`: $\rho_\{\text\{seq\}\} = e^\{0.1\} \approx 1.105$ (vì mean = 0.1).
* `average`: trung bình cộng ~ 1.155.
* `geometric`: $\sqrt\{0.819 \times 1.492\} \approx 1.105$.

Đây là lúc `sequence` và `geometric` thể hiện variance reduction.

---

## 13. Tuning thực tế

* **Bắt đầu với `token` (mặc định)**: an toàn, dễ debug.
* **Nếu rollout > 500 token**: thử `sequence`. Nếu `probs_corr` cải thiện -> giữ.
* **Nếu rollout > 1000 token**: thử `average` hoặc `geometric_average` để balance.
* **Không combine**: chỉ chọn một level; combine 2 level có thể gây counter-intuitive behavior.

---

## 14. Tóm tắt

| Level | Công thức | Variance | Bias | Khi nào |
| --- | --- | --- | --- | --- |
| `token` | $\exp(\delta_t)$ | Cao | 0 | Rollout ngắn |
| `sequence` | $\exp(\frac\{1\}\{T\} \sum \delta_t)$ | Thấp | Hơi biased | Rollout dài, dense reward cuối |
| `average` | $(\rho_t + \rho_\{\text\{seq\}\})/2$ | Trung bình | Trung bình | Rollout trung bình |
| `geometric_average` | $\sqrt\{\rho_t \cdot \rho_\{\text\{seq\}\}\}$ | Trung bình | Trung bình | Mix dễ/khó |

Code: `src/art/loss.py`, đoạn `if importance_sampling_level != "token":`.

---

Tiếp theo: [Theory 4: KL Estimators](theory_4_kl_penalty) - chi tiết về KL penalty và các estimator k1/k2/k3/tito.
