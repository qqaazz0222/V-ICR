# Exporter 모듈

`modules/exporter.py`

## 개요

**Exporter** 모듈은 `metadata.json`과 `recognition_results.json`을 조합하여 최종 행동 라벨 데이터를 생성합니다. 생성된 데이터는 프레임 단위의 바운딩 박스와 행동 라벨을 포함하며, `data/output/` 디렉토리에 비디오 이름으로 저장됩니다.

## 핵심 기능

- **데이터 통합**: 추적 메타데이터 + 인식 결과 조합
- **프레임 단위 라벨링**: 각 프레임에 bbox와 행동 라벨 매핑
- **비디오 레벨 라벨**: 영상 전체의 대표 행동 자동 추출
- **Labelmap 포함**: 행동 ID와 이름 매핑 정보
- **통계 요약**: 행동 분포 및 전체 통계

---

## 함수: `export_action_labels`

### 시그니처

```python
def export_action_labels(
    video_path: str, 
    working_dir: str, 
    output_dir: str
) -> Dict[str, Any]
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `video_path` | str | 원본 비디오 경로 |
| `working_dir` | str | 작업 디렉토리 (tubes/ 포함) |
| `output_dir` | str | 출력 디렉토리 |

### 반환값

최종 라벨 데이터 딕셔너리 (JSON으로도 저장됨)

---

## 출력 파일 구조

### 저장 위치

```
data/output/<video_name>.json
```

### JSON 구조

```json
{
  "version": "1.0",
  "video": {
    "name": "demo",
    "path": "./data/input/demo.mp4",
    "fps": 120.08,
    "width": 1920,
    "height": 1080,
    "total_frames": 1521,
    "duration": 12.67
  },
  "labelmap": {
    "0": {
      "name": "attacking",
      "original_actions": ["punching", "throwing punch", ...]
    },
    "1": {
      "name": "boxing practice",
      "original_actions": ["boxing", "sparring", ...]
    }
  },
  "num_persons": 3,
  "persons": {
    "id_1": {
      "id": "id_1",
      "start_frame": 0,
      "end_frame": 1520,
      "start_time": 0.0,
      "end_time": 12.66,
      "tube_size": {"width": 358, "height": 511},
      "action_summary": ["defending", "standing and moving"],
      "action_timeline": {...},
      "frames": [...]
    }
  },
  "summary": {
    "total_action_instances": 36,
    "action_distribution": {
      "standing and moving": 3,
      "boxing practice": 16
    }
  },
  "video_action": {
    "primary_action": "boxing practice",
    "primary_action_id": 1,
    "primary_percentage": 44.4,
    "top_actions": [
      {"action": "boxing practice", "count": 16, "percentage": 44.4},
      {"action": "defending", "count": 3, "percentage": 8.3}
    ],
    "description": "This video primarily shows 'boxing practice' (44.4% of all detected actions)"
  }
}
```

---

## 주요 필드 설명

### video

비디오 메타데이터

| 필드 | 타입 | 설명 |
|------|------|------|
| `name` | string | 비디오 파일명 (확장자 제외) |
| `fps` | float | 초당 프레임 수 |
| `width` | int | 비디오 가로 해상도 |
| `height` | int | 비디오 세로 해상도 |
| `total_frames` | int | 전체 프레임 수 |
| `duration` | float | 비디오 길이 (초) |

### labelmap

행동 ID → 행동 이름 매핑

```json
{
  "0": {
    "name": "boxing practice",
    "original_actions": ["boxing", "sparring", "shadow boxing"]
  }
}
```

- `name`: 대표 행동 이름
- `original_actions`: 그룹화된 원본 행동들

### persons

사람별 추적 및 행동 데이터

#### 기본 정보

| 필드 | 설명 |
|------|------|
| `id` | 사람 ID (예: "id_1") |
| `start_frame` | 등장 시작 프레임 |
| `end_frame` | 등장 종료 프레임 |
| `start_time` | 시작 시간 (초) |
| `end_time` | 종료 시간 (초) |
| `tube_size` | 튜브 영상 크기 |

#### action_summary

해당 사람이 수행한 모든 행동 목록 (중복 제거)

```json
["defending", "standing and moving", "boxing practice"]
```

#### action_timeline

초 단위 최종 행동 (최고 확률 행동만 저장)

```json
{
  "0": {"action": "standing", "action_id": 15},
  "1": {"action": "boxing practice", "action_id": 1}
}
```

| 필드 | 설명 |
|------|------|
| `action` | 해당 초의 최종 행동 이름 |
| `action_id` | 행동 ID (labelmap 참조) |

#### frames

**프레임 단위** 상세 데이터 - 가장 중요한 필드

```json
[
  {
    "frame_idx": 0,
    "timestamp": 0.0,
    "bbox": {"x1": 1006, "y1": 157, "x2": 1275, "y2": 599},
    "action": "standing and moving",
    "action_id": 15
  },
  {
    "frame_idx": 1,
    "timestamp": 0.0083,
    "bbox": {"x1": 1005, "y1": 157, "x2": 1274, "y2": 599},
    "action": "standing and moving",
    "action_id": 15
  }
]
```

| 필드 | 설명 |
|------|------|
| `frame_idx` | 프레임 인덱스 |
| `timestamp` | 타임스탬프 (초) |
| `bbox` | 바운딩 박스 좌표 |
| `action` | 행동 이름 |
| `action_id` | 행동 ID (labelmap 참조) |

### summary

전체 통계 정보

```json
{
  "total_action_instances": 36,
  "action_distribution": {
    "boxing practice": 16,
    "defending": 3,
    "standing and moving": 3
  }
}
```

---

## video_action 필드 (NEW)

영상 전체에 대한 대표 행동 정보

```json
{
  "video_action": {
    "primary_action": "boxing practice",
    "primary_action_id": 1,
    "primary_percentage": 44.4,
    "top_actions": [
      {"action": "boxing practice", "count": 16, "percentage": 44.4},
      {"action": "defending", "count": 3, "percentage": 8.3},
      {"action": "standing", "count": 2, "percentage": 5.6}
    ],
    "description": "This video primarily shows 'boxing practice' (44.4% of all detected actions)"
  }
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `primary_action` | string | 가장 빈번한 행동 |
| `primary_action_id` | int | labelmap에서의 행동 ID (-1: 미등록) |
| `primary_percentage` | float | 전체 행동 중 비율 (%) |
| `top_actions` | array | 상위 5개 행동 (빈도순) |
| `description` | string | 영상 행동 요약 설명 |

### 활용 예시

```python
# 비디오 레벨 행동 확인
video_action = data["video_action"]
print(f"주요 행동: {video_action['primary_action']}")
print(f"비율: {video_action['primary_percentage']}%")

# 상위 행동들
for action in video_action["top_actions"]:
    print(f"  - {action['action']}: {action['count']}회 ({action['percentage']}%)")
```

---

## 사용 예시

### 기본 사용

```python
from modules.exporter import export_action_labels

result = export_action_labels(
    video_path="./data/input/demo.mp4",
    working_dir="./data/working/demo",
    output_dir="./data/output"
)

print(f"저장됨: data/output/{result['video']['name']}.json")
print(f"인원 수: {result['num_persons']}")
print(f"행동 분포: {result['summary']['action_distribution']}")
```

### 프레임별 라벨 접근

```python
import json

with open("data/output/demo.json", "r") as f:
    data = json.load(f)

# 특정 사람의 프레임별 행동 확인
person = data["persons"]["id_1"]
for frame in person["frames"][:10]:
    print(f"Frame {frame['frame_idx']}: {frame['action']} @ {frame['bbox']}")
```

### 행동 통계 분석

```python
# 행동별 빈도
for action, count in data["summary"]["action_distribution"].items():
    print(f"{action}: {count}회")

# 특정 사람의 주요 행동
person = data["persons"]["id_1"]
print(f"주요 행동: {person['action_summary']}")
```

---

## 데이터 흐름

```
┌───────────────────────────────────────────────────────────┐
│  입력                                                      │
 │                                                           │
│  working/<video>/tubes/metadata.json                       │
│  → 프레임별 bbox, 트랙 정보                                 │
 │                                                           │
│  working/<video>/tubes/recognition_results.json            │
│  → 초별 행동 라벨, labelmap                                 │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  export_action_labels()                                    │
 │                                                           │
│  1. 비디오 메타데이터 추출 (fps, 해상도)                    │
│  2. Labelmap 구성                                          │
│  3. 프레임 → 초 매핑 (frame_idx / fps = second)            │
│  4. 각 프레임에 해당 초의 행동 라벨 할당                    │
│  5. 통계 계산                                              │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  출력                                                      │
 │                                                           │
│  output/<video>.json                                       │
│  → 프레임 단위 bbox + 행동 라벨 통합 데이터                 │
└───────────────────────────────────────────────────────────┘
```

---

## 활용 사례

### 1. 시각화 도구 개발

```python
# 영상에 bbox와 행동 라벨 오버레이
for person_id, person in data["persons"].items():
    for frame_info in person["frames"]:
        draw_bbox(frame_info["bbox"], frame_info["action"])
```

### 2. 학습 데이터 생성

```python
# 행동 인식 모델 학습용 데이터 변환
for person in data["persons"].values():
    for frame in person["frames"]:
        save_training_sample(
            video=data["video"]["path"],
            frame_idx=frame["frame_idx"],
            bbox=frame["bbox"],
            label=frame["action_id"]
        )
```

### 3. 행동 분석 리포트

```python
# 시간대별 행동 분석
timeline = {}
for person in data["persons"].values():
    for sec, info in person["action_timeline"].items():
        timeline.setdefault(int(sec), []).append({
            "person": person["id"],
            "action": info["action"]
        })
```

---

## 커맨드라인 사용

```bash
python modules/exporter.py <video_path> <working_dir> <output_dir>

# 예시
python modules/exporter.py ./data/input/demo.mp4 ./data/working/demo ./data/output
```
