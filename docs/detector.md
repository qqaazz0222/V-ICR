# Detector 모듈

`modules/detector.py`

## 개요

**Detector** 클래스는 비디오에서 사람을 탐지하고 추적하여 개인별 "튜브(tube)"를 추출하는 모듈입니다. YOLO12와 ByteTrack 알고리즘을 기반으로 하며, Kalman 필터 스무딩, 트랙 스티칭, **CoTracker 궤적 추출** 등의 후처리를 수행합니다.

### 주요 기능

- **사람 탐지 및 추적**: YOLO12 + ByteTrack
- **트랙 후처리**: Kalman 필터 스무딩, 끊어진 트랙 스티칭
- **CoTracker 궤적**: 사람별 포인트 궤적 추출 및 필터링
- **GPU 메모리 최적화**: 처리 후 자동 메모리 정리

## 클래스: Detector

### 초기화

```python
Detector(model_path="./checkpoints/yolo12x.pt", min_tube_length=30)
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `model_path` | str | `./checkpoints/yolo12x.pt` | YOLO 모델 가중치 경로 |
| `min_tube_length` | int | 30 | 유효한 튜브로 인정되는 최소 프레임 수 |

### 주요 메서드

#### `infer(video_path, save_dir)`

메인 추론 파이프라인을 실행합니다.

```python
detector = Detector()
detector.infer("./data/input/video.mp4", "./data/working/video")
```

**파라미터:**
- `video_path` (str): 입력 비디오 경로
- `save_dir` (str): 결과 저장 디렉토리

**생성 결과:**
```
save_dir/
 ├─ detect/        # 탐지 결과 시각화 (선택적)
 ├─ track/         # 추적 결과 시각화
└── tubes/         # 추출된 튜브들
     ├─ ─ id_1.mp4
     ├─ ─ id_2.mp4
    └─ ─ metadata.json
```

---

## 내부 처리 파이프라인

### 1단계: 탐지 (`_detect`)

YOLO12를 사용하여 각 프레임에서 사람(class 0)을 탐지합니다.

```python
self.model.predict(
    source=video_path,
    classes=[0],  # Person only
    ...
)
```

### 2단계: 추적 (`_track`)

ByteTrack 알고리즘으로 프레임 간 사람 ID를 유지합니다.

**사용 설정 파일:** `modules/bytetrack_tuned.yaml`

```python
self.model.track(
    source=video_path,
    tracker="modules/bytetrack_tuned.yaml",
    persist=True,
    classes=[0],
    conf=0.6,
    ...
)
```

### 3단계: CoTracker 궤적 추출 (`_run_cotracker`)

**CoTrackerOnlinePredictor**를 사용하여 비디오 전체의 포인트 궤적을 추출합니다.

**특징:**
- 온라인 모드: 비디오를 청크 단위로 처리하여 메모리 효율성 향상
- 자동 체크포인트 다운로드: `scaled_online.pth`
- GPU 메모리 관리: 처리 후 자동 정리

```python
def _run_cotracker(self, video_path, device='cuda'):
    # 1. 체크포인트 다운로드 (없으면)
    checkpoint_path = f"./checkpoints/scaled_online.pth"
    self._download_cotracker_checkpoint(checkpoint_path)
    
    # 2. 모델 로드
    model = CoTrackerOnlinePredictor(checkpoint_path)
    model = model.to(device)
    
    # 3. 비디오 청크 처리
    # step 간격으로 프레임을 배치 처리
    
    # 4. 결과: (T, N, 2) 궤적, (T, N) 가시성
    return tracks, visibility
```

**출력:**
- `tracks`: (T, N, 2) - T프레임, N개 포인트, xy 좌표
- `visibility`: (T, N) - 각 포인트 가시성

### 4단계: 튜브 추출 (`_extract_tubes`)

추적 결과에서 개인별 튜브를 추출합니다. 이 과정에서 다음 후처리가 적용됩니다:

#### 3.1 트랙 스티칭 (Broken Track Stitching)

끊어진 트랙을 자동으로 연결합니다.

**조건:**
- 시간 갭: 최대 3초 (90프레임 @ 30fps)
- 공간 거리: 최대 100픽셀 (중심점 기준)

**연결 방식:**
- 갭 구간에 선형 보간(Linear Interpolation)된 바운딩 박스 생성

```python
# 갭 프레임에 보간된 박스 추가
for f_idx in range(prev_end + 1, curr_start):
    alpha = (f_idx - prev_end) / (curr_start - prev_end)
    interp_box = box_start * (1 - alpha) + box_end * alpha
```

#### 3.2 Kalman 필터 스무딩

바운딩 박스의 지터(떨림)를 제거합니다.

**상태 벡터:** `[cx, cy, w, h, vx, vy, vw, vh]`
- 위치 (cx, cy): 박스 중심 좌표
- 크기 (w, h): 박스 너비, 높이
- 속도 (vx, vy, vw, vh): 각 요소의 변화율

**노이즈 파라미터:**
```python
# Process Noise (모델 불확실성)
process_noise = [0.0005, 0.0005, 0.0001, 0.0001,  # 위치/크기
                 0.001, 0.001, 0.0005, 0.0005]     # 속도

# Measurement Noise (측정 불확실성)  
measurement_noise = [100.0, 100.0,  # 위치
                     200.0, 200.0]  # 크기
```

> 💡 **설계 의도**: 측정 노이즈를 높게 설정하여 모델(예측)을 더 신뢰하고, 스무딩 효과를 극대화합니다.

#### 3.3 종횡비 조정 및 레터박스

각 튜브의 프레임을 일관된 크기로 추출합니다.

```python
def adjust_bbox_to_aspect(box, target_aspect, img_w, img_h):
    # 목표 종횡비에 맞게 박스 확장
    ...

def letterbox_resize(img, target_size):
    # 레터박스 방식으로 리사이즈 (검은색 패딩)
    ...
```

---

## 출력: metadata.json

각 튜브의 메타데이터를 JSON 형식으로 저장합니다.

```json
{
  "id_1": {
    "start_frame": 0,
    "end_frame": 150,
    "width": 120,
    "height": 280,
    "bboxes": [
      {"frame_idx": 0, "box": [100, 50, 220, 330]},
      {"frame_idx": 1, "box": [102, 51, 222, 331]},
      ...
    ]
  },
  "id_2": {
    ...
  }
}
```

| 필드 | 설명 |
|------|------|
| `start_frame` | 트랙 시작 프레임 (0-indexed) |
| `end_frame` | 트랙 종료 프레임 |
| `width` | 튜브 비디오 너비 (최대 박스 기준) |
| `height` | 튜브 비디오 높이 (최대 박스 기준) |
| `bboxes` | 프레임별 바운딩 박스 [x1, y1, x2, y2] |

---

## ByteTrack 설정

`modules/bytetrack_tuned.yaml` 파일로 추적 동작을 커스터마이즈합니다.

```yaml
tracker_type: bytetrack
track_high_thresh: 0.5   # 1단계 매칭 임계값
track_low_thresh: 0.1    # 2단계 저신뢰도 매칭
new_track_thresh: 0.6    # 새 트랙 생성 임계값
track_buffer: 60         # 손실된 트랙 유지 프레임
match_thresh: 0.8        # IoU 연관 임계값
fuse_score: True         # 탐지 점수 융합
```

### 파라미터 튜닝 가이드

| 목표 | 조정 방법 |
|------|----------|
| 더 적은 오탐 | `new_track_thresh` ↑ |
| 더 긴 트랙 유지 | `track_buffer` ↑ |
| ID 전환 감소 | `match_thresh` ↓ |
| 저신뢰도 탐지 활용 | `track_low_thresh` ↓ |

---

## 사용 예시

### 기본 사용

```python
from modules.detector import Detector

detector = Detector()
detector.infer("./video.mp4", "./output")
```

### 커스텀 설정

```python
detector = Detector(
    model_path="./my_model.pt",
    min_tube_length=60  # 최소 2초 (30fps 기준)
)
detector.infer("./video.mp4", "./output")
```

---

## 주의사항

1. **GPU 메모리**: YOLO12x는 약 4-6GB VRAM 필요
2. **출력 덮어쓰기**: `infer()` 호출 시 기존 `track/`, `tubes/` 디렉토리 삭제됨
3. **비디오 코덱**: 출력 튜브는 `mp4v` 코덱 사용 (일부 플레이어 호환성 주의)
