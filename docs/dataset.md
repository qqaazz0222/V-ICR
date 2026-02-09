# Dataset 모듈

`modules/dataset.py`

## 개요

**Dataset** 모듈은 입력 비디오 파일을 로드하고 관리하는 유틸리티를 제공합니다. `VideoDataset` 클래스와 `VideoInfo` 데이터클래스를 통해 비디오 파일 경로, 작업 디렉토리, 출력 디렉토리를 체계적으로 관리합니다.

## 핵심 기능

- **비디오 검색**: 입력 디렉토리에서 비디오 파일 자동 검색
- **다중 형식 지원**: MP4, AVI, MOV, MKV, WebM 등
- **디렉토리 관리**: 작업 디렉토리 및 출력 디렉토리 자동 생성
- **이터레이터 지원**: for 루프를 통한 비디오 순회

---

## 데이터클래스: VideoInfo

### 정의

```python
@dataclass
class VideoInfo:
    path: str         # 비디오 파일 경로
    name: str         # 비디오 이름 (확장자 제외)
    working_dir: str  # 작업 디렉토리 경로
    tubes_dir: str    # 튜브 디렉토리 경로
```

### 사용 예시

```python
video = VideoInfo(
    path="./data/input/demo.mp4",
    name="demo",
    working_dir="./data/working/demo",
    tubes_dir="./data/working/demo/tubes"
)
```

---

## 클래스: VideoDataset

### 초기화

```python
VideoDataset(
    input_dir="./data/input",
    working_dir="./data/working",
    output_dir="./data/output"
)
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `input_dir` | str | `./data/input` | 입력 비디오 디렉토리 |
| `working_dir` | str | `./data/working` | 작업 디렉토리 |
| `output_dir` | str | `./data/output` | 출력 디렉토리 |

### 지원 확장자

```python
SUPPORTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
```

---

## 주요 메서드

### `load(extensions=None) -> int`

입력 디렉토리에서 비디오 파일을 검색합니다.

**파라미터:**
- `extensions`: 검색할 확장자 목록 (기본값: 지원 확장자 전체)

**반환값:**
- 발견된 비디오 수

```python
dataset = VideoDataset()
num_videos = dataset.load()
print(f"Found {num_videos} videos")
```

### `__len__() -> int`

비디오 수를 반환합니다.

```python
print(len(dataset))  # 3
```

### `__iter__()`

비디오 순회 이터레이터를 반환합니다.

```python
for video in dataset:
    print(f"Processing: {video.name}")
```

### `__getitem__(index) -> VideoInfo`

인덱스로 비디오 정보에 접근합니다.

```python
first_video = dataset[0]
print(first_video.path)
```

---

## 프로퍼티

| 프로퍼티 | 반환 타입 | 설명 |
|---------|----------|------|
| `videos` | List[VideoInfo] | 모든 비디오 목록 |
| `is_empty` | bool | 비디오가 없는지 여부 |

---

## 편의 함수

### `load_videos(input_dir, working_dir, output_dir, extensions) -> VideoDataset`

비디오 데이터셋을 로드하는 편의 함수입니다.

```python
from modules.dataset import load_videos

dataset = load_videos("./data/input")
for video in dataset:
    print(video.name)
```

---

## 사용 예시

### 기본 사용

```python
from modules.dataset import VideoDataset

# 데이터셋 초기화 및 로드
dataset = VideoDataset()
num_videos = dataset.load()

if dataset.is_empty:
    print("No videos found")
    exit(0)

# 비디오 순회
for video in dataset:
    print(f"Video: {video.name}")
    print(f"  Path: {video.path}")
    print(f"  Working dir: {video.working_dir}")
    print(f"  Tubes dir: {video.tubes_dir}")
```

### 커스텀 디렉토리

```python
dataset = VideoDataset(
    input_dir="/path/to/videos",
    working_dir="/path/to/work",
    output_dir="/path/to/output"
)
dataset.load()
```

### 특정 확장자만 검색

```python
dataset = VideoDataset()
dataset.load(extensions=[".mp4", ".avi"])
```

---

## run.py에서의 사용

```python
from modules.dataset import VideoDataset

dataset = VideoDataset(
    input_dir=args.input_dir,
    working_dir=args.working_dir,
    output_dir=args.output_dir
)
num_videos = dataset.load()

with logger.progress_bar(len(dataset), "Processing") as progress_ctx:
    for video in dataset:
        detector.infer(video.path, video.working_dir)
        recognizer.recognize(video.path, video.tubes_dir)
        export_action_labels(video.path, video.working_dir, dataset.output_dir)
        progress_ctx.advance()
```

---

## 디렉토리 구조

```
data/
 ├─ input/              # 입력 비디오 (사용자가 배치)
 │   ├─ ─ video1.mp4
 │   ├─ ─ video2.mp4
 │  └─ ─ label_map.txt   # 사전 정의된 라벨맵 (선택사항)
 ├─ working/            # 작업 디렉토리 (자동 생성)
 │   ├─ ─ video1/
 │   │   ├─ ─ tubes/
 │   │   │   ├─ ─ id_1.mp4
 │   │   │   ├─ ─ metadata.json
 │   │   │  └─ ─ recognition_results.json
 │   │  └─ ─ label_map.txt
 │  └─ ─ video2/
 │      └─ ─ ...
└── output/             # 출력 디렉토리 (자동 생성)
     ├─ ─ video1.json
    └─ ─ video2.json
```
