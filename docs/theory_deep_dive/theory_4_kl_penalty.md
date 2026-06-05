---
sidebar_position: 5
sidebar_label: "Theory 4: KL Penalty & Estimators"
---

# Theory 4: KL Penalty & Estimators: k1, k2, k3, tito

KL divergence giữa policy hiện tại \(\pi_\theta\) và reference \(\pi_{\text{ref}}\) (thường là base model) là thước đo "policy đã lệch bao xa khỏi điểm khởi đầu". Bài này phân tích các estimator, vì sao ART dùng dạng đặc biệt, và ảnh hưởng của nó đến training stability.

---

## 1. KL divergence định nghĩa

\[
KL(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{a \sim \pi_\theta} \left[ \log \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)} \right]
\]

Vấn đề: expectation lấy theo \(\pi_\theta\) (policy hiện tại). Nếu ta chỉ có samples từ \(\pi_{\text{old}}\), cần estimator khác.

Có 4 estimator phổ biến (Schulman 2020):

| Estimator | Công thức | Bias | Variance | Ghi chú |
| --- | --- | --- | --- | --- |
| k1 | \(\log r = \log \pi_\theta - \log \pi_{\text{ref}}\) | Thấp (gần exact) | Cao | Đơn giản nhất |
| k2 | \(0.5 (\log r)^2\) | Trung bình | Trung bình | Symmetric, luôn không âm |
| k3 | \(-\log r\) | Trung bình | Thấp | Trái dấu với k1 |
| tito | \((r - 1) - \log r\) | Thấp | Trung bình | Taylor expansion chính xác bậc 2 |

với \(r = \pi_\theta / \pi_{\text{ref}}\).

---

## 2. Phân tích từng estimator

### 2.1. k1 estimator

\[
\hat{KL}_{k1} = \log r = \log \pi_\theta - \log \pi_{\text{ref}}
\]

Khi \(\pi_\theta = \pi_{\text{ref}}\), \(\hat{KL} = 0\) (đúng). Khi \(\pi_\theta\) lệch xa, \(\hat{KL}\) tăng magnitude. Nhưng có thể âm (nếu \(\pi_\theta < \pi_{\text{ref}}\)).

Variance cao vì:
* Mỗi sample có \(\log r_t\) dao động lớn.
* Khi \(r \to 0\) (token có \(\pi_{\text{ref}}\) lớn nhưng \(\pi_\theta\) nhỏ), \(\log r\) âm rất lớn.
* Khi \(r \to \infty\) (token có \(\pi_{\text{ref}}\) nhỏ nhưng \(\pi_\theta\) lớn), \(\log r\) dương rất lớn.

### 2.2. k2 estimator

\[
\hat{KL}_{k2} = 0.5 (\log r)^2
\]

Luôn không âm, symmetric quanh 0. Variance trung bình. Đây là estimator phổ biến nhất trong practice (vì luôn dương, dễ visualize).

Hạn chế: tại \(\log r = 0\), đạo hàm = 0, không phạt gradient nhỏ. Cần warmup.

### 2.3. k3 estimator

\[
\hat{KL}_{k3} = -\log r
\]

Trái dấu với k1. Variance thấp hơn vì:
* Sample với \(r\) nhỏ -> \(\log r\) âm lớn -> \(-\log r\) dương lớn (cùng dấu với "lệch").
* Sample với \(r\) lớn -> \(-\log r\) âm (sai dấu).

Đây là estimator ART chọn. Lý do: trong RL loss, ta thêm KL penalty vào **advantage**. Nếu dùng k1 hoặc k2, KL luôn dương -> advantage giảm -> policy bị phạt đều. Nếu dùng k3, KL âm cho \(r > 1\) (policy "tự tin hơn" ref) -> advantage tăng -> khuyến khích policy tự tin hơn ref. Có thể gây drift.

### 2.4. tito estimator (token-ITO)

\[
\hat{KL}_{tito} = (r - 1) - \log r
\]

Taylor expansion của KL quanh \(r = 1\):

\[
KL(r) = (r - 1) - \log r + O((r-1)^3)
\]

Tito là bậc 2 chính xác. Bias thấp, variance trung bình. Được dùng trong nhiều paper RLHF gần đây.

Tại \(r = 1\): \(\hat{KL} = 0\). Tại \(r = 2\): \(\hat{KL} = 1 - \log 2 \approx 0.307\). Tại \(r = 0.5\): \(\hat{KL} = -0.5 + \log 2 \approx 0.193\).

Cả hai phía đều dương (vì KL luôn không âm). Nhưng hàm không smooth ở \(r = 0\).

---

## 3. ART dùng estimator nào?

Trong `src/art/loss.py`:

```python
if kl_penalty_coef > 0 and ref_logprobs is not None:
    kl_per_token = (new_logprobs - ref_logprobs).detach() * assistant_mask
    avg_kl = aligned_inputs.masked_mean(kl_per_token, assistant_mask)
    kl_penalty = kl_penalty_coef * (avg_kl - kl_per_token) * assistant_mask
    advantages = advantages + kl_penalty
```

Để ý: `kl_per_token = (new_logprobs - ref_logprobs)` chính là **k1 estimator** (nhưng có dấu dương vì ta subtract: log pi_theta - log pi_ref). Hmm, thực ra:

* `(new - ref) = log pi_theta - log pi_ref = log r = k1` (với r = pi_theta / pi_ref).
* KL divergence định nghĩa là `E[log pi_theta - log pi_ref]` = `E[k1]`. Vậy `kl_per_token` chính là **k1 per token**.

Nhưng ART không dùng `kl_per_token` trực tiếp làm penalty. Thay vào đó:

\[
\text{kl\_penalty}_t = \beta \cdot (\overline{KL} - \text{k1}_t)
\]

với \(\overline{KL}\) = mean của k1 trong rollout.

Đây là dạng **"subtract mean"** trick. Công thức tương đương với việc thêm vào advantage một hằng số âm (\(-\beta \cdot \overline{KL}\)) và trừ đi k1 per-token.

Hệ quả:

* Token có k1 > mean: penalty > 0, advantage tăng (khuyến khích... hmm).

Wait, sign kì lạ. Để ý: `kl_penalty = beta * (avg_kl - kl_per_token)`.

* Nếu `kl_per_token > avg_kl` (token lệch nhiều hơn trung bình): `kl_penalty` âm.
* Nếu `kl_per_token < avg_kl` (token lệch ít hơn trung bình): `kl_penalty` dương.

Vậy token lệch nhiều bị **phạt** (kl_penalty âm -> advantage giảm), token lệch ít được **thưởng** (kl_penalty dương -> advantage tăng). Đây là "relative KL penalty" - so sánh với mean.

Cấu trúc này tương tự **GRPO advantage** (zero-mean trong group): mỗi token so với mean.

---

## 4. Tại sao subtract mean?

Nếu không subtract mean:

* Mọi token bị phạt vì KL > 0.
* Policy cố giữ gần ref.
* Exploration bị hạn chế.

Với subtract mean:

* Token lệch nhiều hơn mean: bị phạt (nhưng có thể là do advantageous update).
* Token lệch ít hơn mean: được thưởng (có thể là do "stable" update).

Hệ quả: policy có thể lệch xa ref ở một số token, miễn là tổng thể không lệch quá. Đây là "soft trust region".

Một cách hiểu khác: \(\beta \cdot (\overline{KL} - \text{k1}_t)\) là một hằng số âm (\(-\beta \cdot \overline{KL}\)) cộng với \(-\beta \cdot \text{k1}_t\). Constant shift không ảnh hưởng gradient (vì nó cộng vào advantage). Vậy phần "có ý nghĩa" là \(-\beta \cdot \text{k1}_t\) - chính là **k3 estimator** scaled bởi \(-\beta\).

Vậy ART thực chất dùng k3 estimator, với scale \(\beta\), trừ đi mean (constant shift bị cancel bởi autograd? Không hẳn - mean được tính trước khi add, nên nó là một hằng số "cộng vào" advantage).

Hmm, thực ra ta cần phân tích gradient. Vì `kl_penalty` được add vào `advantages`, và `advantages` được dùng trong policy loss:

\[
L = -\sum_t \text{clip}(\rho_t) \cdot A_t \cdot \log \pi_\theta(a_t|s_t)
\]

\[
\frac{\partial L}{\partial \theta} = -\sum_t \text{clip}(\rho_t) \cdot A_t \cdot \frac{\partial \log \pi_\theta}{\partial \theta}
\]

KL penalty làm thay đổi \(A_t\). Constant shift (như \(-\beta \cdot \overline{KL}\)) được áp dụng cho **mọi** token, sẽ bị cancel trong GRPO advantage (vì GRPO đã zero-mean). Nhưng trong CISPO không có zero-mean automatic; constant shift làm thay đổi gradient magnitude.

Vậy subtract mean thực sự có tác dụng: nó làm KL penalty **zero-mean** theo rollout, tránh làm lệch gradient magnitude.

---

## 5. Kết hợp với GRPO

GRPO advantage đã zero-mean trong group. KL penalty thêm constant shift (qua mean) sẽ bị cancel. Phần còn lại \(-\beta \cdot \text{k1}_t\) (k3 estimator) làm thay đổi relative advantage giữa các token.

Ví dụ:

* Group 4 rollout, 100 token mỗi rollout.
* GRPO advantage: A_i trong [-2, 2], zero-mean.
* KL penalty: k1_t = 0.05 trung bình, dao động từ -0.1 đến +0.2.
* Sau KL penalty: A_i tăng/giảm từ -0.05 đến +0.025 (với beta = 0.5).

Vẫn zero-mean trong group (vì mean của \(-\beta \cdot \text{k1}_t\) qua group là 0). GRPO chuẩn hóa lại.

---

## 6. Variance reduction

Vì k3 estimator = \(-\log r = \log \pi_{\text{ref}} - \log \pi_\theta\) = **negative k1**, variance của k3 = variance của k1. Vậy nó không thấp hơn k1 về variance.

ART giảm variance bằng **subtract mean**: \(\overline{KL} - \text{k1}_t\) có mean = 0, std thấp hơn k1 đơn lẻ. Cụ thể:

\[
\text{Var}(\overline{KL} - \text{k1}_t) = \text{Var}(\text{k1}_t) \cdot (1 - 1/T)
\]

Với T = 100, giảm 1% variance. Không nhiều. Lợi ích chính của subtract mean là **zero-mean** (cancel constant shift trong GRPO).

---

## 7. Kimi K2 tau: một estimator khác

Trong `loss.py`:

```python
if tau := experimental_config.get("kimi_k2_tau", None):
    advantages = advantages - tau * logprob_diff.detach()
```

`logprob_diff = new_logprobs - old_logprobs` (giữa policy hiện tại và old, không phải ref). Đây là một "self-regularization": nếu policy mới lệch nhiều khỏi old, advantage bị giảm.

Đây là estimator giống k3 nhưng so với \(\pi_{\text{old}}\) thay vì \(\pi_{\text{ref}}\). Lợi: không cần ref model, free. Hại: không đảm bảo "không quên" base knowledge (chỉ ổn định quanh old).

Tổ hợp `kimi_k2_tau` + `kl_penalty_coef` cho cả hai mức regularization:
* `tau`: ngắn hạn (giữ gần old).
* `beta`: dài hạn (giữ gần ref).

---

## 8. Khi nào tăng/giảm `kl_penalty_coef`

| Tình huống | Hành động |
| --- | --- |
| `probs_corr` giảm nhanh (< 0.85) | Tăng `kl_penalty_coef` lên 0.05-0.1 |
| Policy "quên" kiến thức base | Tăng `kl_penalty_coef` |
| `kl_policy_ref` tăng đều | Tốt; reward vẫn tăng nghĩa là ref không kìm hãm |
| `kl_policy_ref > 0.5` | Cảnh báo; policy quá lệch ref |
| Training chậm, reward không tăng | Giảm `kl_penalty_coef` xuống 0.001 hoặc 0 |
| Domain sensitive (medical, legal) | Tăng `kl_penalty_coef` cao (0.1-0.5) |

Trong ART, metric `kl_policy_ref` được log tự động (xem `Loss.kl_policy_ref`).

---

## 9. Kết hợp với reference model

Reference model \(\pi_{\text{ref}}\) mặc định là base model (Qwen, Llama). Trong LocalBackend, reference = base model loaded riêng (cho ref_logprobs).

Vấn đề: forward qua ref model tốn memory và compute. ART dùng **shared base**: ref model và policy chia sẻ base weights, chỉ khác adapter (LoRA). Khi train LoRA, ref = base + frozen old LoRA.

Trong `loss.py`, `ref_logprobs` được tính sẵn ở đâu đó trước khi gọi `loss_fn`. Nó là tensor có shape `[batch, seq]`, đã được shift tương ứng.

---

## 10. Numerical stability

`logprob_diff = new_logprobs - ref_logprobs` có thể rất lớn (nếu model xác suất cực thấp). Trong k3 estimator, \(-\log r\) có thể overflow. ART không clip `kl_per_token`, nhưng `kl_penalty_coef` thường nhỏ (0.001-0.1) nên effect lên advantage vẫn bounded.

Nếu gặp overflow, kiểm tra:

* Có token nào có `ref_logprobs = -inf` không (token chưa từng xuất hiện trong vocab của ref model).
* `new_logprobs` có quá nhỏ không (`< -1000`).

Workaround: clamp `logprobs` ở `[-100, 0]`.

---

## 11. So sánh các estimator qua bảng

| Estimator | Công thức gốc | ART dùng | Vai trò |
| --- | --- | --- | --- |
| k1 | \(\log r\) | Có (qua `(new - ref)`) | Direct KL |
| k2 | \(0.5(\log r)^2\) | Không | Symmetric |
| k3 | \(-\log r\) | Có (qua subtract mean) | Negative feedback |
| tito | \((r-1) - \log r\) | Không | Taylor approx |
| kimi_k2_tau | \(-\tau \cdot \log(\pi_\theta/\pi_{\text{old}})\) | Có (qua `kimi_k2_tau`) | Self-regularization |

---

## 12. Ví dụ số: 1 token

* `new_logprobs = -2.0` (tức \(\pi_\theta = e^{-2} = 0.135\)).
* `ref_logprobs = -3.0` (tức \(\pi_{\text{ref}} = e^{-3} = 0.050\)).
* `log r = -2 - (-3) = 1.0` -> \(r = e \approx 2.718\).

KL estimators:

* k1: \(\log r = 1.0\)
* k2: \(0.5 \times 1 = 0.5\)
* k3: \(-1.0\)
* tito: \((2.718 - 1) - 1 = 0.718\)

ART's k3 + subtract mean: giả sử mean KL của rollout = 0.5. `kl_penalty_t = beta * (0.5 - 1.0) = -0.5 * beta`. Token này bị phạt 0.5*beta.

Nếu `beta = 0.05`, kl_penalty = -0.025. Token advantage giảm 0.025.

---

## 13. Tuning thực tế

Bắt đầu với:

```python
config = dev.TrainConfig(
    kl_penalty_coef=0.01,
    kimi_k2_tau=None,
)
```

* Nếu training ổn định, reward tăng: giữ.
* Nếu reward tăng chậm: giảm `kl_penalty_coef` xuống 0.001.
* Nếu policy drift quá nhanh: tăng lên 0.05 hoặc kết hợp `kimi_k2_tau=0.1`.
* Nếu cần regularization mạnh (medical, legal): dùng `kl_penalty_coef=0.1` + `kimi_k2_tau=0.5`.

---

## 14. Tóm tắt

| Khía cạnh | ART cách tiếp cận |
| --- | --- |
| Estimator | k3 qua `-(new - ref)` |
| Aggregation | Per-token, sau đó subtract mean trong rollout |
| Hyperparameter | `kl_penalty_coef` (mặc định 0) |
| Tích hợp với advantage | Cộng thẳng vào advantages |
| Backward | `detach` kl_per_token -> không có gradient qua new_logprobs |
| Kết hợp với `kimi_k2_tau` | Self-regularization với old policy |
| Metric log | `kl_policy_ref` (mean) |

Code: `src/art/loss.py`, khối `if kl_penalty_coef > 0 and ref_logprobs is not None:`.

---

Tiếp theo: [Theory 5: RULER Math](theory_5_ruler_math) - toán học đằng sau relative LLM-as-judge.
