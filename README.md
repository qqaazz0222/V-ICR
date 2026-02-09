# V-ICR: Video Iterative Context Refinement

**비디오 기반 행동 인식 파이프라인** - YOLO + ByteTrack 추적과 Qwen3-VL 기반 반복적 맥락 정제를 통한 행동 인식 시스템

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[🇺🇸 English README](docs/README_EN.md)

## 📋 개요

V-ICR은 비디오에서 사람을 탐지·추적하고, 각 사람의 행동을 Vision-Language Model(VLM)을 활용하여 인식하는 엔드투엔드 파이프라인입니다.

### 주요 특징

- **🎯 정밀한 사람 추적**: YOLO12 + ByteTrack 기반 강건한 멀티오브젝트 추적
- **🔧 트랙 후처리**: Kalman 필터 스무딩, 끊어진 트랙 스티칭
- **📍 CoTracker 궤적 추적**: 사람별 포인트 궤적 추출 및 필터링
- **🧠 VLM 기반 행동 인식**: Qwen3-VL-8B를 활용한 1초 단위 행동 분류
- **🔍 Phase 0 Action Discovery**: 대표 튜브 샘플링 → 자동 labelmap 생성
- **🔄 시간적 soft-label 정제**: Soft-label 후보군과 시간적 맥락을 활용한 반복 정제
- **🎬 비디오 레벨 라벨**: 영상 전체에 대한 대표 행동 자동 추출
- **⚡ GPU 메모리 최적화**: 비디오별 메모리 정리 및 캐싱

## 🏗️ 시스템 아키텍처

```
입력 비디오 (MP4)
        │
        ▼
┌───────────────────┐
 │   Detector       │ ← YOLO12 + ByteTrack + CoTracker
│  (탐지 & 추적)      │
│  - 사람 추적       │
│  - 포인트 궤적 추출  │
└─────────┬─────────┘
          │
          ▼
    ┌───────────┐
     │  Tubes   │ ← 개인별 크롭 비디오
    └─ ────┬─────┘
          │
          ▼
┌───────────────────┐
 │  Recognizer      │ ← Qwen3-VL-8B
│  (행동 인식)        │
 │                  │
│  Phase 0: 대표 튜브 샘플링 & 어휘 발견 │
│  Phase 1: Labelmap 기반 분류  │
│  Phase 2: 반복 정제           │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
 │   Exporter       │
│  (라벨 데이터 생성)   │
│  - 프레임별 라벨    │
│  - 비디오 레벨 라벨  │
└─────────┬─────────┘
          │
          ▼
  output/<video>.json
```

## 📁 프로젝트 구조

```
V-ICR/
 ├─ run.py                    # 메인 실행 스크립트
 ├─ requirements.txt          # Python 의존성
 ├─ checkpoints/              # 모델 가중치
 │  └─ ─ yolo12x.pt           # YOLO12 가중치
 ├─ modules/                  # 핵심 모듈
 │   ├─ ─ detector.py          # 탐지 및 추적 모듈
 │   ├─ ─ recognizer.py        # 행동 인식 모듈
 │   ├─ ─ exporter.py          # 라벨 데이터 출력 모듈
 │   ├─ ─ dataset.py           # 데이터셋 유틸리티
 │  └─ ─ bytetrack_tuned.yaml # ByteTrack 설정
 ├─ utils/                    # 유틸리티
 │  └─ ─ logger.py            # 로깅 유틸리티
 ├─ data/                     # 데이터 디렉토리
 │   ├─ ─ input/               # 입력 비디오 (여기에 MP4 파일 배치)
 │   ├─ ─ working/             # 중간 결과물
 │  └─ ─ output/              # 최종 라벨 데이터 출력
└── docs/                     # 문서
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 준비

YOLO12 가중치를 `checkpoints/` 디렉토리에 배치합니다.
Qwen3-VL-8B는 첫 실행 시 자동으로 HuggingFace에서 다운로드됩니다.

### 3. 비디오 처리

```bash
# 입력 디렉토리에 비디오 배치
cp your_video.mp4 data/input/

# 처리 실행
python run.py
```

### 4. 사전 정의된 라벨맵 (선택사항)

`data/input/label_map.txt` 파일에 사전 정의된 행동 라벨을 지정할 수 있습니다.
이 파일이 존재하고 라벨이 있으면, 인식 단계에서 해당 라벨만 사용합니다.

```
# label_map.txt 예시 (두 가지 형식 모두 지원)

# 형식 1: 라벨만
standing
walking
running
punching
blocking

# 형식 2: 번호: 라벨
0: standing
1: walking
2: running
```

### 5. 결과 확인

**중간 결과물** (`data/working/<video_name>/`):
- `tubes/` - 개인별 크롭 비디오
- `tubes/metadata.json` - 추적 메타데이터
- `tubes/recognition_results.json` - 행동 인식 결과
- `label_map.txt` - 행동 카테고리 매핑

**최종 출력** (`data/output/`):
- `<video_name>.json` - 프레임 단위 통합 라벨 데이터

## ⚙️ 명령줄 옵션

```bash
python run.py [OPTIONS]

옵션:
  --skip-recognition          행동 인식 단계 스킵 (탐지만 수행)
  --refinement-iterations N   정제 반복 횟수 (기본값: 2)
  --input-dir DIR             입력 비디오 디렉토리 (기본값: ./data/input)
  --working-dir DIR           작업 디렉토리 (기본값: ./data/working)
  --output-dir DIR            출력 디렉토리 (기본값: ./data/output)
  --dataset NAME              데이터셋 이름 (하위 디렉토리 생성)
```

**예시:**

```bash
# 탐지만 수행
python run.py --skip-recognition

# 정제 3회 반복
python run.py --refinement-iterations 3

# 커스텀 디렉토리 사용
python run.py --input-dir ./my_videos --output-dir ./my_output
```

## 📊 출력 형식

### 최종 라벨 데이터 (`output/<video>.json`)

프레임 단위의 bbox와 행동 라벨을 포함하는 통합 데이터:

```json
{
  "version": "1.0",
  "video": {
    "name": "demo",
    "fps": 120.08,
    "width": 1920,
    "height": 1080,
    "total_frames": 1521,
    "duration": 12.67
  },
  "labelmap": {
    "0": {"name": "attacking", "original_actions": ["punching", ...]},
    "1": {"name": "boxing practice", "original_actions": [...]}
  },
  "num_persons": 3,
  "persons": {
    "id_1": {
      "action_summary": ["defending", "boxing practice"],
      "action_timeline": {
        "0": {"action": "standing", "action_id": 15}
      },
      "frames": [
        {
          "frame_idx": 0,
          "timestamp": 0.0,
          "bbox": {"x1": 1006, "y1": 157, "x2": 1275, "y2": 599},
          "action": "standing and moving",
          "action_id": 15
        }
      ]
    }
  },
  "summary": {
    "total_action_instances": 36,
    "action_distribution": {"boxing practice": 16, "defending": 3}
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

### video_action 필드

| 필드 | 설명 |
|------|------|
| `primary_action` | 영상에서 가장 빈번한 행동 |
| `primary_action_id` | labelmap에서의 행동 ID |
| `primary_percentage` | 전체 행동 중 비율 (%) |
| `top_actions` | 상위 5개 행동 (빈도순) |
| `description` | 영상 행동 요약 설명 |

### label_map.txt

유사 행동 그룹화 결과:

```
0: attacking
   -> punching, throwing punch, raising arms
1: boxing practice
   -> boxing, sparring, shadow boxing
2: defending
   -> blocking, defending, preparing to block
```

### recognition_results.json (중간 결과)

```json
{
  "video_path": "./data/input/demo.mp4",
  "labelmap": ["attacking", "boxing practice", "defending", ...],
  "action_groups": {
    "boxing practice": ["boxing", "sparring", "shadow boxing"]
  },
  "tubes": {
    "id_1": {
      "temporal_labels": {
        "0": [
          {"action": "standing", "confidence": 0.90},
          {"action": "looking", "confidence": 0.75}
        ]
      },
      "final_actions": [
        {"time": 0, "action": "standing", "confidence": 0.90}
      ]
    }
  },
  "id_time_actions": [
    {"id": "id_1", "time": 0, "action": "standing", "confidence": 0.90}
  ]
}
```

## 🔧 설정

### ByteTrack 파라미터 (`modules/bytetrack_tuned.yaml`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `track_high_thresh` | 0.5 | 1단계 매칭 임계값 |
| `track_low_thresh` | 0.1 | 2단계 저신뢰도 매칭 |
| `new_track_thresh` | 0.6 | 새 트랙 생성 임계값 |
| `track_buffer` | 60 | 손실된 트랙 유지 프레임 |
| `match_thresh` | 0.8 | IoU 기반 연관 임계값 |

## 📚 모듈 문서

자세한 모듈별 문서는 `docs/` 디렉토리를 참조하세요:

- [detector.md](docs/detector.md) - 탐지 및 추적 모듈
- [recognizer.md](docs/recognizer.md) - 행동 인식 모듈
- [exporter.md](docs/exporter.md) - 라벨 데이터 출력 모듈
- [dataset.md](docs/dataset.md) - 데이터셋 유틸리티
- [bytetrack_config.md](docs/bytetrack_config.md) - ByteTrack 설정
- [logger.md](docs/logger.md) - 로깅 유틸리티

## 📦 의존성

- Python >= 3.8
- PyTorch >= 2.1.0
- ultralytics >= 8.0.0
- transformers >= 4.38.0
- OpenCV
- Rich (로깅)
- Qwen-VL-Utils

## 🔍 성능 고려사항

- **GPU 메모리**: Qwen3-VL-8B 로딩에 약 16GB VRAM 필요
- **처리 시간**: 1분 비디오 기준 약 5-7분 소요 (GPU 환경, iterations=5)
- **튜브 최소 길이**: 30프레임 미만 트랙은 필터링됨

## 📄 라이선스

MIT License

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
