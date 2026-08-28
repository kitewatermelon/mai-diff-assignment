# mai-ass

MAI 과제 — 레이어별로 일반 self-attention과 Differential Attention을 섞을 수 있는 ViT.

- `model.py` — ViT 정의. `Attention`(일반), `DiffAttention`([Differential Transformer](https://arxiv.org/abs/2410.05258)), 그리고 블록별 선택 로직(`resolve_modes`).
- `main.py` — 모델 생성과 `--mode` CLI 파싱.
- `visualization_attention.py` — 마지막 블록의 어텐션 맵을 png로 저장.

## 설치

```bash
pip install -r requirements.txt
```

CUDA 빌드 torch가 필요하면 [pytorch.org](https://pytorch.org)에서 먼저 설치하세요.

`opencv-python-headless`를 쓰는 이유는, GUI 버전(`opencv-python`)이 딸려오는 Qt 플러그인을 matplotlib이 잡으면서 디스플레이가 없는 환경(WSL, 서버)에서 크래시가 나기 때문입니다. 스크립트에서도 `matplotlib.use('Agg')`로 백엔드를 고정해 둡니다.

## mode

각 transformer 블록이 어떤 어텐션을 쓸지 지정합니다.

| 값 | 의미 |
|----|------|
| `sa` | 일반 self-attention (기본값) |
| `da` | Differential Attention |

블록 인덱스는 `0`부터 `depth-1`까지이며, vit_tiny/small/base 모두 `depth=12`이므로 `0~11`입니다.

### CLI

```bash
python main.py --mode sa                    # 전부 일반 어텐션 (기본값)
python main.py --mode da                    # 전부 differential

# 12개 전부 나열 (개수가 depth와 다르면 에러)
python main.py --mode sa,sa,sa,sa,sa,sa,da,da,da,da,da,da

# 인덱스 지정 — 적지 않은 블록은 sa
python main.py --mode 1:da,4:da,11:da

# 인덱스 완전 열거
python main.py --mode 0:sa,1:da,2:sa,3:da,4:sa,5:da,6:sa,7:da,8:sa,9:da,10:sa,11:sa
```

`i:mode` 형식과 그냥 나열하는 형식은 섞어 쓸 수 없습니다.

실행하면 블록 구성과 파라미터 수를 출력하고 더미 forward로 동작을 확인합니다.

```
$ python main.py --arch vit_small --mode 1:da,4:da,11:da
vit_small patch16 img224 on cuda
blocks: ['sa', 'da', 'sa', 'sa', 'da', 'sa', 'sa', 'sa', 'sa', 'sa', 'sa', 'da']
params: 21.67M
forward: (2, 384) | last attn (da): {'attn1': (2, 6, 197, 197), 'attn2': (2, 6, 197, 197), 'diff': (2, 6, 197, 197)}
```

### 파이썬

CLI 문자열 외에 리스트 · 딕셔너리 · 함수도 받습니다.

```python
from main import build_model

build_model('vit_small', mode='da')                              # 전부 da
build_model('vit_small', mode=['sa'] * 6 + ['da'] * 6)           # 블록별 리스트 (길이 == depth)
build_model('vit_small', mode={0: 'sa', 1: 'da', ..., 11: 'sa'}) # 인덱스 전부 열거
build_model('vit_small', mode={'default': 'sa', -1: 'da'})       # 기본값 + 일부만 지정
build_model('vit_small', mode=lambda i: 'da' if i % 2 else 'sa') # 인덱스 함수
```

- 딕셔너리는 음수 인덱스가 됩니다 (`-1` = 마지막 블록).
- `default` 키가 없으면 `0~depth-1`을 모두 적어야 하고, 빠진 인덱스가 있으면 에러로 알려줍니다.
- 잘못된 이름·개수·범위는 **모델 생성 시점에** `ValueError`로 잡힙니다.
- 확정된 구성은 `model.modes`에 리스트로 남습니다.

## 어텐션 맵

```python
attn = net.get_last_selfattention(x)   # 마지막 블록의 어텐션
```

마지막 블록의 mode에 따라 반환 타입이 다릅니다.

- `sa` → 텐서 `(B, num_heads, N, N)`, 행 합이 1
- `da` → 딕셔너리 `{'attn1', 'attn2', 'diff'}`, 각각 `(B, num_heads, N, N)`

`attn1`/`attn2`는 두 그룹의 softmax 맵이고, 실제로 값에 곱해지는 것은 `diff = attn1 - λ·attn2`입니다. `diff`는 음수를 포함하고 행 합이 1이 아니므로 시각화할 때는 컬러맵을 `vmin=-vmax`로 대칭 설정하세요.

## 어텐션 시각화

`visualization_attention.py`는 이미지 한 장을 넣고 마지막 블록의 어텐션 맵을 png로 저장합니다.

```bash
python visualization_attention.py --arch vit_tiny --mode da \
    --image_path img.png --image_size 224 --threshold 0.5
```

결과는 `output/`에 저장됩니다 (`--output_dir`로 변경 가능). 입력 이미지를 정규화한 사본도 `output/img.png`로 같이 저장되니, `--output_dir`을 입력 이미지가 들어 있는 폴더로 지정하면 원본을 덮어쓸 수 있습니다.

| 인자 | 설명 |
|------|------|
| `--arch` | `vit_tiny` / `vit_small` / `vit_base` |
| `--patch_size` | 패치 크기 (기본 8) |
| `--mode` | 위와 동일한 mode 스펙 |
| `--image_path` | 입력 이미지. 생략하면 DINO 예제 이미지를 내려받습니다 |
| `--image_size` | 리사이즈 크기 |
| `--threshold` | 어텐션 질량의 상위 xx%만 남긴 마스크도 함께 저장 |
| `--output_dir` | 저장 위치 (기본 `output/`) |

출력 파일은 **마지막 블록의 mode**에 따라 달라집니다.

- `sa` → `attn-head{j}.png`
- `da` → `attn1-head{j}.png`, `attn2-head{j}.png`, `diff-head{j}.png` (헤드마다 3장)

`--threshold`를 주면 각 맵마다 `mask_th{값}_{이름}-head{j}.png`도 나옵니다.

`diff`는 음수를 포함하므로 0을 중앙에 둔 발산형 컬러맵(`bwr`)으로 그립니다 — 빨강이 양수, 파랑이 음수입니다. 마스크는 누적 질량 기준이라 음수에는 정의되지 않아서 `diff`의 양수 부분만 사용합니다 (실행 시 안내 메시지가 출력됩니다).

> mode에 `da`가 섞여 있어도 **마지막 블록이 `sa`이면** 맵은 한 장만 나옵니다.

## 구현 메모

- `DiffAttention`의 head_dim은 `dim // num_heads // 2`입니다. 같은 `num_heads`에서도 차원이 절반인 쿼리/키를 두 세트 사용하므로 `dim`이 `2 * num_heads`로 나누어떨어져야 합니다.
- `lambda_init = 0.8 - 0.6 * exp(-0.3 * depth)`의 `depth`는 전체 기준 블록 인덱스입니다. sa 블록을 섞어도 실제 레이어 위치를 따릅니다.
- `DiffAttention`은 `qk_scale`을 받지 않고 `head_dim ** -0.5`를 직접 계산합니다.
