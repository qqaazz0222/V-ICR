# Logger 유틸리티

`utils/logger.py`

## 개요

**Logger** 클래스는 V-ICR 프로젝트의 로깅 유틸리티입니다. [Rich](https://github.com/Textualize/rich) 라이브러리를 사용하여 깔끔하고 가독성 높은 콘솔 출력을 제공합니다.

## 클래스: Logger

### 초기화

```python
from utils.logger import Logger

logger = Logger()
```

## 주요 메서드

### 기본 출력

| 메서드 | 설명 | 예시 출력 |
|--------|------|----------|
| `print(*args)` | 기본 출력 | 일반 텍스트 |
| `print_info(msg)` | 정보 메시지 | `[>] Found 3 videos` |
| `print_success(msg)` | 성공 메시지 (녹색) | `[+] Processing complete` |
| `print_warning(msg)` | 경고 메시지 (노란색) | `[!] Low confidence` |
| `print_error(msg)` | 에러 메시지 (빨간색) | `[-] File not found` |

### 헤더 출력

```python
logger.print_header("V-ICR Video Processor")
```

**출력:**
```
╔════════════════════════════════════════╗
║  V-ICR Video Processor                 ║
╚════════════════════════════════════════╝
```

### 처리 결과 출력

```python
logger.print_item_result(
    name="video1",
    success=True,
    elapsed_time=45.2
)
```

**성공 시:**
```
    [+] video1 (45.2s)
```

**실패 시:**
```python
logger.print_item_result(
    name="video2",
    success=False,
    elapsed_time=12.3,
    error="Memory overflow"
)
```
```
    [-] video2 FAILED: Memory overflow
```

### 요약 출력

```python
results = [
    {"name": "video1", "status": "success", "time": 45.2},
    {"name": "video2", "status": "failed", "time": 12.3}
]
logger.print_summary(results, total_time=57.5)
```

**출력:**
```
== Summary ==
  Total: 2 | Success: 1 | Failed: 1 | Time: 57.5s
[!] 1 item(s) failed.
```

---

## 컨텍스트 매니저

### Status Spinner

작업 진행 중 스피너 애니메이션을 표시합니다.

```python
with logger.status("Loading model..."):
    model = load_heavy_model()
logger.print_success("Model loaded")
```

**출력:**
```
⠋ Loading model...    (애니메이션)
[+] Model loaded
```

### Progress Bar

진행률 막대를 표시합니다.

```python
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]

with logger.progress_bar(len(videos), "Processing") as progress:
    for video in videos:
        progress.update_description(f"Processing: {video}")
        process_video(video)
        progress.advance()
```

**출력:**
```
⠧ Processing: video2.mp4 ━━━━━━━━━━━━━ 66% | 0:01:30 | 0:00:45
```

#### ProgressContext 메서드

| 메서드 | 설명 |
|--------|------|
| `advance(amount=1)` | 진행률 증가 |
| `update_description(desc)` | 설명 텍스트 업데이트 |

---

## 헬퍼 함수

### format_time(seconds)

초를 사람이 읽기 쉬운 형식으로 변환합니다.

```python
from utils.logger import format_time

format_time(45.3)      # "45.3s"
format_time(125.7)     # "2m 5.7s"
format_time(3725.2)    # "1h 2m 5.2s"
```

### get_logger()

싱글톤 패턴으로 기본 로거 인스턴스를 반환합니다.

```python
from utils.logger import get_logger

logger = get_logger()  # 항상 같은 인스턴스 반환
```

---

## 사용 예시

### 전체 워크플로우

```python
from utils.logger import Logger, format_time

logger = Logger()

# 헤더
logger.print_header("My Application")

# 초기화
logger.print_info("Starting initialization...")
with logger.status("Loading resources..."):
    time.sleep(2)
logger.print_success("Initialization complete")

# 처리
items = ["item1", "item2", "item3"]
results = []

with logger.progress_bar(len(items), "Processing") as progress:
    for item in items:
        progress.update_description(f"Processing: {item}")
        
        try:
            process(item)
            results.append({"name": item, "status": "success", "time": 10.0})
            logger.print_item_result(item, success=True, elapsed_time=10.0)
        except Exception as e:
            results.append({"name": item, "status": "failed", "time": 5.0})
            logger.print_item_result(item, success=False, elapsed_time=5.0, error=str(e))
        
        progress.advance()

# 요약
logger.print_summary(results, total_time=30.0)
```

---

## Rich 스타일 참조

Logger는 Rich 마크업을 지원합니다:

```python
logger.print("[bold]Bold text[/bold]")
logger.print("[red]Red text[/red]")
logger.print("[green]Green[/green] and [blue]Blue[/blue]")
```

자세한 스타일은 [Rich 문서](https://rich.readthedocs.io/en/stable/markup.html)를 참조하세요.

---

## 주의사항

1. **Rich 의존성**: `pip install rich` 필요
2. **ANSI 지원**: 터미널이 ANSI 색상을 지원해야 함
3. **너비 자동 조정**: 터미널 너비에 맞게 출력 조정
