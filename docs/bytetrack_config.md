# ByteTrack 설정 가이드

`modules/bytetrack_tuned.yaml`

## 개요

ByteTrack은 V-ICR의 사람 추적에 사용되는 다중 객체 추적(MOT) 알고리즘입니다. 이 문서는 `bytetrack_tuned.yaml` 설정 파일의 각 파라미터와 튜닝 방법을 설명합니다.

## 현재 설정

```yaml
tracker_type: bytetrack

# 매칭 임계값
track_high_thresh: 0.5   # 1단계 매칭 임계값
track_low_thresh: 0.1    # 2단계 저신뢰도 매칭
new_track_thresh: 0.6    # 새 트랙 생성 임계값

# 트랙 관리
track_buffer: 60         # 손실된 트랙 유지 프레임
match_thresh: 0.8        # IoU 기반 연관 임계값

# 점수 융합
fuse_score: True         # 탐지 점수와 IoU 융합
```

## 파라미터 상세

### tracker_type

| 값 | 설명 |
|---|------|
| `bytetrack` | ByteTrack 알고리즘 사용 |
| `botsort` | BoT-SORT 알고리즘 사용 (Re-ID 지원) |

V-ICR은 `bytetrack`을 사용합니다. Re-ID가 필요한 경우 `botsort`로 변경할 수 있습니다.

### track_high_thresh

**1단계 매칭의 탐지 신뢰도 임계값**

- **범위**: 0.0 ~ 1.0
- **기본값(ultralytics)**: 0.25
- **V-ICR 설정**: 0.5

| 값 | 효과 |
|---|------|
| 높음 (0.5+) | 높은 신뢰도 탐지만 매칭 → 깨끗한 트랙, 일부 놓침 |
| 낮음 (0.2-) | 저신뢰도 탐지도 매칭 → 더 많은 트랙, 오탐 증가 |

> 💡 **V-ICR 선택 이유**: `conf=0.6` 탐지 설정과 조화를 이루면서 안정적인 트랙 생성

### track_low_thresh

**2단계 매칭의 탐지 신뢰도 임계값**

- **범위**: 0.0 ~ track_high_thresh
- **기본값(ultralytics)**: 0.1
- **V-ICR 설정**: 0.1

ByteTrack의 핵심 아이디어: 1단계에서 매칭되지 않은 저신뢰도 탐지를 2단계에서 재시도

| 값 | 효과 |
|---|------|
| 높음 | 2단계 후보 감소 → 보수적 매칭 |
| 낮음 | 매우 낮은 신뢰도도 시도 → 가림 상황에서 복구력 증가 |

### new_track_thresh

**새로운 트랙 생성 임계값**

- **범위**: 0.0 ~ 1.0
- **기본값(ultralytics)**: 0.25
- **V-ICR 설정**: 0.6

기존 트랙과 매칭되지 않은 탐지가 새 트랙이 되기 위한 최소 신뢰도

| 값 | 효과 |
|---|------|
| 높음 (0.6) | 확실한 탐지만 새 트랙 생성 → ID 수 안정 |
| 낮음 (0.2) | 저신뢰도도 새 트랙 생성 → ID 폭발 위험 |

> 💡 **V-ICR 선택 이유**: `detector.py`의 `conf=0.6`과 일치시켜 일관성 유지

### track_buffer

**손실된 트랙 유지 프레임 수**

- **범위**: 1 ~ ∞
- **기본값(ultralytics)**: 30
- **V-ICR 설정**: 60

트랙이 탐지되지 않아도 "Lost" 상태로 유지되는 프레임 수

| 값 | 효과 |
|---|------|
| 높음 (60+) | 긴 가림 후에도 ID 유지 → ID 전환 감소 |
| 낮음 (15-) | 빠른 트랙 종료 → 메모리 효율, ID 전환 증가 |

> 💡 **V-ICR 선택 이유**: 30fps 기준 2초간 유지, `detector.py`의 3초 스티칭과 조화

### match_thresh

**IoU 기반 연관 임계값**

- **범위**: 0.0 ~ 1.0
- **기본값(ultralytics)**: 0.8
- **V-ICR 설정**: 0.8

예측 위치와 탐지 박스 간 최소 IoU

| 값 | 효과 |
|---|------|
| 높음 (0.8) | 엄격한 매칭 → 정확하지만 빠른 움직임에서 실패 |
| 낮음 (0.5) | 관대한 매칭 → 급격한 움직임 대응, 오매칭 위험 |

### fuse_score

**탐지 점수 융합 여부**

- **타입**: Boolean
- **기본값(ultralytics)**: True
- **V-ICR 설정**: True

탐지 신뢰도를 IoU 매칭 점수에 융합

| 값 | 효과 |
|---|------|
| True | 높은 신뢰도 탐지 우선 매칭 → 안정적인 추적 |
| False | 순수 IoU 기반 → 모든 탐지 동등 취급 |

## 시나리오별 설정 예시

### 혼잡한 장면 (많은 사람)

```yaml
track_high_thresh: 0.6   # 더 엄격한 매칭
new_track_thresh: 0.7    # 확실한 탐지만 새 트랙
track_buffer: 30         # 짧은 버퍼로 ID 폭발 방지
match_thresh: 0.7        # 약간 관대한 매칭
```

### 가림이 많은 장면

```yaml
track_high_thresh: 0.4   # 저신뢰도도 매칭 시도
track_low_thresh: 0.05   # 2단계에서도 적극적
track_buffer: 90         # 긴 가림 대응
```

### 빠른 움직임

```yaml
match_thresh: 0.6        # 관대한 IoU 매칭
track_buffer: 45         # 중간 정도 버퍼
```

### 최소 오탐 (정밀 우선)

```yaml
track_high_thresh: 0.7
new_track_thresh: 0.8
match_thresh: 0.85
track_buffer: 20
```

## Detector.py와의 상호작용

`detector.py`는 추가적인 후처리를 수행하여 ByteTrack의 한계를 보완합니다:

| ByteTrack 한계 | Detector 보완 |
|---------------|---------------|
| ID 전환 | 3초 이내 갭 스티칭 |
| 박스 떨림 | Kalman 필터 스무딩 |
| 짧은 트랙 | `min_tube_length=30` 필터링 |

## 디버깅 팁

1. **트랙이 너무 자주 끊어짐**
   - `track_buffer` ↑
   - `match_thresh` ↓

2. **ID가 너무 많이 생성됨**
   - `new_track_thresh` ↑
   - `track_high_thresh` ↑

3. **트랙이 다른 사람에게 점프**
   - `match_thresh` ↑
   - `fuse_score: True` 확인

4. **가린 사람이 복구되지 않음**
   - `track_low_thresh` ↓
   - `track_buffer` ↑
