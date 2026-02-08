# Recognizer 모듈

`modules/recognizer.py`

## 개요

**Recognizer** 클래스는 추출된 사람 튜브에서 행동을 인식하는 VLM(Vision-Language Model) 기반 모듈입니다. Qwen3-VL-8B 모델을 사용하며, **1초 단위 시간별 분석**, **유사 행동 그룹화**, **labelmap 기반 정제**를 수행합니다.

## 핵심 특징

- **1초 단위 분석**: 튜브 영상을 1초 세그먼트로 분할하여 분석
- **Soft-Label**: 각 초마다 5개의 행동 후보군 (action, confidence 쌍)
- **유사 행동 그룹화**: Phase 1 후 유사한 행동들을 대표 행동으로 그룹화
- **Labelmap 기반 정제**: 그룹화된 labelmap 내의 행동만으로 확률값 판단
- **최종 출력**: `id-time-action` 형식의 정제된 결과

## 클래스: Recognizer

### 초기화

```python
Recognizer(model_name="Qwen/Qwen3-VL-8B-Instruct")
```

### 주요 메서드

#### `recognize(video_path, tubes_dir, iterations=2, predefined_labelmap=None)`

메인 인식 파이프라인을 실행합니다.

```python
recognizer = Recognizer()

# 기본 사용 (자동 그룹화)
result = recognizer.recognize(
    video_path="./data/input/video.mp4",
    tubes_dir="./data/working/video/tubes",
    iterations=2
)

# 사전 정의된 라벨맵 사용
result = recognizer.recognize(
    video_path="./data/input/video.mp4",
    tubes_dir="./data/working/video/tubes",
    iterations=2,
    predefined_labelmap=["standing", "walking", "punching", "blocking"]
)
```

**파라미터:**

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `video_path` | str | 원본 비디오 경로 |
| `tubes_dir` | str | 튜브 디렉토리 (metadata.json 포함) |
| `iterations` | int | 정제 반복 횟수 (기본값: 2) |
| `predefined_labelmap` | List[str] | 사전 정의된 라벨 리스트 (선택사항) |

> **Note**: `predefined_labelmap`이 제공되면 Phase 1에서부터 해당 라벨만 사용하고, Phase 2 (자동 그룹화)는 스킵됩니다.

---

## 파이프라인 구조

### 기본 동작 (자동 그룹화)

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 초기 분석 (1초 단위)                                    │
│                                                                  │
│  모든 튜브에 대해:                                                │
│  [0초] → 5개 후보 → ["standing", "looking", "waiting", ...]      │
│  [1초] → 5개 후보                                                 │
│  ...                                                             │
│                                                                  │
│  모든 감지된 행동 수집                                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 유사 행동 그룹화 & Labelmap 생성                        │
│                                                                  │
│  VLM을 통해 유사 행동 그룹화:                                     │
│  - "sitting down" + "standing up" 반복 → "squatting"             │
│  - "walking forward" + "moving" → "walking"                      │
│                                                                  │
│  → label_map.txt 저장 (working 디렉토리)                          │
│  → 5~10개의 대표 행동 카테고리                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Labelmap 기반 반복 정제                                 │
│                                                                  │
│  매 초마다:                                                       │
│  - 이전/다음 초 컨텍스트                                          │
│  - Labelmap 내 행동에 대해서만 확률값 판단                         │
│                                                                  │
│  반복 횟수: iterations - 1                                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: 최종 id-time-action 추출                               │
│                                                                  │
│  각 초의 top-1 후보 → {id, time, action, confidence}             │
└─────────────────────────────────────────────────────────────────┘
```

### 사전 정의된 Labelmap 사용 시

`data/input/label_map.txt`에 라벨 목록이 존재하거나 `predefined_labelmap` 파라미터가 전달되면:

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 초기 분석 (사전 정의된 라벨만 사용)                       │
│                                                                  │
│  VLM에 "이 라벨 중에서만 선택하세요" 라는 제약 조건 제공            │
│  → 처음부터 labelmap 내의 행동만 후보로 생성                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 스킵됨                                                 │
│                                                                  │
│  그룹화 불필요 - 이미 정의된 라벨 사용                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    (Phase 3 → Phase 4 동일)
```

**장점:**
- 일관된 라벨링: 여러 비디오에서 동일한 라벨 사용 보장
- 빠른 처리: VLM 그룹화 단계 스킵
- 도메인 특화: 특정 도메인의 행동 어휘만 사용

---

## 유사 행동 그룹화

### 예시

**Phase 1에서 감지된 행동들:**
```
sitting down, standing up, sitting, rising, walking, moving forward, 
looking around, observing, turning head, ...
```

**VLM 그룹화 결과:**
```json
{
  "squatting": ["sitting down", "standing up", "sitting", "rising"],
  "walking": ["walking", "moving forward"],
  "looking": ["looking around", "observing", "turning head"]
}
```

### 그룹화 규칙

1. 동일하거나 매우 유사한 움직임은 그룹화
2. 반복 패턴(앉기+일어서기)은 복합 행동명 사용 (예: "squatting")
3. 명확히 다른 행동은 분리 유지
4. 최종 5~10개 카테고리 목표

---

## Labelmap

### 저장 위치

`data/working/<video_name>/label_map.txt`

### 파일 형식

```
0: squatting
   -> sitting down, standing up, sitting, rising
1: walking
   -> walking, moving forward
2: looking
   -> looking around, observing, turning head
3: punching
   -> punching, throwing punch
```

---

## 출력 형식

### recognition_results.json

```json
{
  "video_path": "./data/input/demo.mp4",
  "labelmap": ["looking", "punching", "squatting", "walking"],
  "action_groups": {
    "squatting": ["sitting down", "standing up"],
    "walking": ["walking", "moving forward"],
    "punching": ["punching", "throwing punch"]
  },
  "tubes": {
    "id_1": {
      "tube_id": "id_1",
      "duration": 10,
      "temporal_labels": {
        "0": [
          {"action": "squatting", "confidence": 0.90},
          {"action": "walking", "confidence": 0.60},
          ...
        ]
      },
      "final_actions": [
        {"time": 0, "action": "squatting", "confidence": 0.90}
      ]
    }
  },
  "id_time_actions": [
    {"id": "id_1", "time": 0, "action": "squatting", "confidence": 0.90},
    {"id": "id_1", "time": 1, "action": "walking", "confidence": 0.88}
  ],
  "action_vocabulary": ["looking", "punching", "squatting", "walking"]
}
```

### 주요 필드

| 필드 | 설명 |
|------|------|
| `labelmap` | 최종 행동 레이블 목록 |
| `action_groups` | 대표 행동 → 원본 행동 매핑 |
| `tubes.<id>.temporal_labels` | 초별 soft-label (labelmap 기준) |
| `id_time_actions` | 최종 id-time-action 목록 |

---

## 내부 메서드

### `_group_similar_actions(all_actions, video_path)`

VLM을 사용하여 유사한 행동을 그룹화합니다.

**입력:**
- `all_actions`: Phase 1에서 감지된 모든 행동 리스트
- `video_path`: 컨텍스트용 비디오 경로

**출력:**
```python
{
    "squatting": ["sitting down", "standing up"],
    "walking": ["walking", "moving"]
}
```

### `_create_labelmap(action_groups, save_dir)`

그룹화 결과를 label_map.txt로 저장합니다.

### `_map_action_to_label(action, action_groups)`

원본 행동 이름을 대표 레이블로 매핑합니다.

### `_refine_second_with_labelmap(frames, candidates, labelmap, prev_action, next_action)`

**Labelmap 내의 행동만** 사용하여 확률값을 재산정합니다.

---

## 사용 예시

### 기본 사용

```python
from modules.recognizer import Recognizer

recognizer = Recognizer()
result = recognizer.recognize(
    "./data/input/video.mp4",
    "./data/working/video/tubes"
)

# Labelmap 확인
print(result["labelmap"])
# ['looking', 'punching', 'squatting', 'walking']

# 행동 그룹 확인
for rep, originals in result["action_groups"].items():
    print(f"{rep}: {originals}")
```

### Labelmap 파일 확인

```bash
cat data/working/video/label_map.txt
```

---

## 성능 고려사항

| 단계 | VLM 호출 횟수 | 설명 |
|------|--------------|------|
| Phase 1 | 튜브 수 × 초 | 초기 분석 |
| Phase 2 | 1회 | 그룹화 |
| Phase 3 | 튜브 수 × 초 × (iterations-1) | 정제 |

**예상 처리 시간 (10초 튜브 3개, iterations=2):**
- Phase 1: ~90초
- Phase 2: ~5초
- Phase 3: ~90초
- 총: ~3분

---

## 이전 버전과의 차이

| 항목 | 이전 | 현재 |
|------|------|------|
| 행동 어휘 | 무제한 | Labelmap으로 제한 |
| 그룹화 | 없음 | VLM 기반 자동 그룹화 |
| 정제 방식 | 전체 어휘 사용 | Labelmap 내 행동만 |
| 출력 파일 | recognition_results.json | + label_map.txt |