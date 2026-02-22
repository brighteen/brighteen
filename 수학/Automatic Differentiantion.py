"""
XOR 문제 신경망 — 순전파 (Forward Propagation)

네트워크 구조:
    입력(2) → 은닉층(2, sigmoid) → 출력(1) → MSE 손실

    * 편향을 가중치 행렬의 첫 번째 행으로 통합
      → 입력 벡터 앞에 1을 붙여(augment) 한 번의 행렬곱으로 선형 변환 + 편향 처리
      → 역전파 시 가중치·편향 그래디언트를 하나의 행렬로 한꺼번에 계산 가능
    * 입력 데이터 중 4번째 샘플 [1, 1]만 사용
"""

import numpy as np

# ── 활성화 함수 ──────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ── 데이터 준비 ──────────────────────────────────────────
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])          # 입력 (4×2)

T = np.array([[0],
              [1],
              [1],
              [0]])             # 타깃 (4×1)

idx = 3                         # 사용할 샘플 인덱스
x = X[idx]                      # [1, 1]
t = T[idx]                      # [0]

# 편향용 1을 앞에 붙여 확장 입력 벡터 생성: [1, x0, x1]
x_aug = np.insert(x, 0, 1)       # (3,)

print(f"[데이터]  입력: {x},  확장 입력: {x_aug},  타깃: {t}")

# ── 가중치 초기화 (편향 포함) ────────────────────────────
#   각 행렬의 첫 번째 행이 편향(bias) 역할
#
#   W1 (3×2): bias 1 + 입력 2  →  은닉 2
#       [ b_h0      b_h1     ]   ← 편향
#       [ w_x0→h0   w_x0→h1 ]
#       [ w_x1→h0   w_x1→h1 ]
#
#   W2 (3×1): bias 1 + 은닉 2  →  출력 1
#       [ b_o    ]               ← 편향
#       [ w_h0→o ]
#       [ w_h1→o ]

np.random.seed(0)

W1 = np.random.rand(3, 2)      # (3×2)  입력+편향 → 은닉
W2 = np.random.rand(3, 1)      # (3×1)  은닉+편향 → 출력

print(f"\n[W1] (입력+편향 → 은닉):\n{W1}")
print(f"[W2] (은닉+편향 → 출력):\n{W2}")

# ── 순전파 ───────────────────────────────────────────────
# 1) 은닉층: 선형 변환 → 활성화
y_hidden = x_aug @ W1                   # (3,)@(3×2) = (2,)
z_hidden = sigmoid(y_hidden)            # (2,)

print(f"\n[은닉층] 선형 출력:  {y_hidden}")
print(f"[은닉층] 활성화 출력: {z_hidden}")

# 2) 출력층: 활성화 출력 앞에 1을 붙여 선형 변환
z_aug = np.insert(z_hidden, 0, 1)       # (3,)
y_out  = z_aug @ W2                     # (3,)@(3×1) = (1,)

print(f"[출력층] 출력: {y_out[0]:.6f}")

# ── 손실 계산 (MSE) ──────────────────────────────────────
loss = (y_out[0] - t[0]) ** 2

print(f"\n[손실]   MSE = {loss:.6f}")


# ══════════════════════════════════════════════════════════
#  역전파 (Backpropagation) — 1회
# ══════════════════════════════════════════════════════════
#
#  순전파 계산 그래프를 거꾸로 따라가며, 연쇄 법칙(chain rule)으로
#  ∂Loss/∂W1, ∂Loss/∂W2 를 한 단계씩 직접 계산한다.
#
#  순전파 흐름 (요약):
#    x_aug ──W1──→ y_hidden ──σ──→ z_hidden ──W2──→ y_out ──MSE──→ loss
#
#  역전파 흐름 (역순):
#    dL/dloss → dL/dy_out → dL/dW2
#                         → dL/dz_hidden → dL/dy_hidden → dL/dW1
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 56)
print("  역전파: 각 단계별 그래디언트")
print("=" * 56)

# ── ① ∂L/∂y_out ─────────────────────────────────────────
#  L = (y_out - t)²
#  ∂L/∂y_out = 2(y_out - t)
dL_dy_out = 2 * (y_out[0] - t[0])              # 스칼라

print(f"\n① ∂L/∂y_out = 2·(y_out - t) = {dL_dy_out:.6f}")

# ── ② ∂L/∂W2 ────────────────────────────────────────────
#  y_out = z_aug · W2   (z_aug = [1, z_h0, z_h1])
#  ∂y_out/∂W2 = z_aug^T
#  ∂L/∂W2 = z_aug^T · ∂L/∂y_out
dL_dW2 = z_aug.reshape(-1, 1) * dL_dy_out      # (3,1)

print(f"\n② ∂L/∂W2 = z_aug^T · ∂L/∂y_out:")
print(f"   W2[0,0] (bias) : {dL_dW2[0,0]:+.6f}")
print(f"   W2[1,0]        : {dL_dW2[1,0]:+.6f}")
print(f"   W2[2,0]        : {dL_dW2[2,0]:+.6f}")

# ── ③ ∂L/∂z_hidden ──────────────────────────────────────
#  y_out = z_aug · W2 에서 z_aug = [1, z_h0, z_h1]
#  ∂y_out/∂z_aug = W2^T
#  ∂L/∂z_aug = W2^T · ∂L/∂y_out  →  z_aug[0]=1은 상수이므로 [1:] 만 필요
dL_dz_aug = W2.flatten() * dL_dy_out           # (3,)
dL_dz_hidden = dL_dz_aug[1:]                   # (2,)  편향 부분 제외

print(f"\n③ ∂L/∂z_hidden = W2[1:]^T · ∂L/∂y_out:")
print(f"   dL/dz_h0 : {dL_dz_hidden[0]:+.6f}")
print(f"   dL/dz_h1 : {dL_dz_hidden[1]:+.6f}")

# ── ④ ∂L/∂y_hidden ──────────────────────────────────────
#  z_hidden = σ(y_hidden)
#  σ'(y) = σ(y)·(1 - σ(y)) = z_hidden · (1 - z_hidden)
#  ∂L/∂y_hidden = ∂L/∂z_hidden ⊙ σ'(y_hidden)     (원소별 곱)
dsigmoid = z_hidden * (1 - z_hidden)                # (2,)
dL_dy_hidden = dL_dz_hidden * dsigmoid              # (2,)

print(f"\n④ σ'(y_hidden) = z·(1-z):")
print(f"   σ'(y_h0) : {dsigmoid[0]:.6f}")
print(f"   σ'(y_h1) : {dsigmoid[1]:.6f}")
print(f"   ∂L/∂y_hidden = ∂L/∂z_hidden ⊙ σ':")
print(f"   dL/dy_h0 : {dL_dy_hidden[0]:+.6f}")
print(f"   dL/dy_h1 : {dL_dy_hidden[1]:+.6f}")

# ── ⑤ ∂L/∂W1 ────────────────────────────────────────────
#  y_hidden = x_aug · W1
#  ∂y_hidden/∂W1 = x_aug^T
#  ∂L/∂W1 = x_aug^T ⊗ ∂L/∂y_hidden     (외적: outer product)
dL_dW1 = x_aug.reshape(-1, 1) @ dL_dy_hidden.reshape(1, -1)  # (3,1)@(1,2) = (3,2)

print(f"\n⑤ ∂L/∂W1 = x_aug^T ⊗ ∂L/∂y_hidden:")
for i in range(3):
    label = " ← bias" if i == 0 else ""
    print(f"   W1[{i},0]={dL_dW1[i,0]:+.6f}   W1[{i},1]={dL_dW1[i,1]:+.6f}{label}")

# ── 결과 요약 ────────────────────────────────────────────
print(f"\n{'═'*56}")
print(f"  그래디언트 요약")
print(f"{'═'*56}")
print(f"\n  ∂L/∂W1 (3×2):\n{dL_dW1}")
print(f"\n  ∂L/∂W2 (3×1):\n{dL_dW2}")