"""
[KOR] V-ICR 프로젝트용 로깅 유틸리티 모듈
[ENG] Logging utility module for V-ICR project

[KOR] Rich 라이브러리를 사용하여 깔끔한 ASCII 기반 콘솔 출력을 제공합니다.
[ENG] Provides clean, ASCII-based console output using Rich library.
"""

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich import box
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


def format_time(seconds: float) -> str:
    """
    [KOR] 초를 사람이 읽기 쉬운 형식으로 변환
    [ENG] Convert seconds to human-readable format
    
    Args:
        seconds: [KOR] 시간 (초) / [ENG] Time in seconds
        
    Returns:
        [KOR] 사람이 읽기 쉬운 시간 문자열
        [ENG] Human-readable time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


class Logger:
    """
    [KOR] Rich를 사용한 ASCII 기반 로깅 클래스
    [ENG] ASCII-based logging class using Rich
    
    [KOR] 프로그레스 바와 테이블을 포함한 깔끔한 콘솔 출력을 제공합니다.
    [ENG] Provides clean console output with progress bars and tables.
    """
    
    def __init__(self):
        """
        [KOR] Logger 초기화
        [ENG] Initialize Logger
        """
        self.console = Console()
    
    def print(self, *args, **kwargs):
        """
        [KOR] 기본 출력
        [ENG] Basic print
        """
        self.console.print(*args, **kwargs)
    
    def print_header(self, title: str):
        """
        [KOR] 프로그램 헤더 출력
        [ENG] Print program header
        
        Args:
            title: [KOR] 헤더 제목 / [ENG] Header title
        """
        self.console.print()
        self.console.print(Panel(
            f"[bold]{title}[/bold]",
            box=box.DOUBLE,
            border_style="white",
            padding=(0, 2)
        ))
        self.console.print()
    
    def print_info(self, message: str):
        """
        [KOR] 정보 메시지 출력
        [ENG] Print info message
        
        Args:
            message: [KOR] 메시지 / [ENG] Message
        """
        self.console.print(f"[>] {message}")
    
    def print_success(self, message: str):
        """
        [KOR] 성공 메시지 출력
        [ENG] Print success message
        
        Args:
            message: [KOR] 메시지 / [ENG] Message
        """
        self.console.print(f"[green][+] {message}[/green]")
    
    def print_warning(self, message: str):
        """
        [KOR] 경고 메시지 출력
        [ENG] Print warning message
        
        Args:
            message: [KOR] 메시지 / [ENG] Message
        """
        self.console.print(f"[yellow][!] {message}[/yellow]")
    
    def print_error(self, message: str):
        """
        [KOR] 에러 메시지 출력
        [ENG] Print error message
        
        Args:
            message: [KOR] 메시지 / [ENG] Message
        """
        self.console.print(f"[red][-] {message}[/red]")
    
    def print_item_result(self, name: str, success: bool, elapsed_time: float, error: Optional[str] = None):
        """
        [KOR] 개별 항목 처리 결과 출력
        [ENG] Print individual item processing result
        
        Args:
            name: [KOR] 항목 이름 / [ENG] Item name
            success: [KOR] 성공 여부 / [ENG] Success status
            elapsed_time: [KOR] 처리 시간 (초) / [ENG] Processing time in seconds
            error: [KOR] 에러 메시지 (실패 시) / [ENG] Error message (if failed)
        """
        if success:
            self.console.print(f"    [green][+] {name}[/green] ({format_time(elapsed_time)})")
        else:
            error_msg = f": {error}" if error else ""
            self.console.print(f"    [red][-] {name}[/red] FAILED{error_msg}")
    
    def print_summary(self, results: List[Dict[str, Any]], total_time: float):
        """
        [KOR] 처리 요약 출력
        [ENG] Print processing summary
        
        Args:
            results: [KOR] 결과 리스트 (각각 name, status, time 키 포함) / [ENG] List of results (each with name, status, time keys)
            total_time: [KOR] 총 처리 시간 / [ENG] Total processing time
        """
        success_count = sum(1 for r in results if r["status"] == "success")
        fail_count = len(results) - success_count
        
        self.console.print()
        self.console.print(f"[bold]== Summary ==[/bold]")
        self.console.print(f"  Total: {len(results)} | Success: [green]{success_count}[/green] | Failed: [red]{fail_count}[/red] | Time: {format_time(total_time)}")
        
        if fail_count == 0:
            self.console.print(f"[green][+] All processing completed successfully.[/green]")
        else:
            self.console.print(f"[yellow][!] {fail_count} item(s) failed.[/yellow]")
    
    @contextmanager
    def status(self, message: str):
        """
        [KOR] 상태 스피너 컨텍스트 매니저
        [ENG] Status spinner context manager
        
        Args:
            message: [KOR] 상태 메시지 / [ENG] Status message
        """
        with self.console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
            yield
    
    @contextmanager
    def progress_bar(self, total: int, description: str = "Progress"):
        """
        [KOR] 프로그레스 바 컨텍스트 매니저
        [ENG] Progress bar context manager
        
        Args:
            total: [KOR] 총 항목 수 / [ENG] Total number of items
            description: [KOR] 프로그레스 설명 / [ENG] Progress description
            
        Yields:
            ProgressContext: [KOR] 프로그레스 업데이트용 컨텍스트 객체 / [ENG] Context object for updating progress
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30, style="cyan", complete_style="green"),
            TaskProgressColumn(),
            TextColumn("|"),
            TimeElapsedColumn(),
            TextColumn("|"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False
        ) as progress:
            task_id = progress.add_task(description, total=total)
            
            class ProgressContext:
                """
                [KOR] 프로그레스 컨텍스트 클래스
                [ENG] Progress context class
                """
                def __init__(self, progress_obj, task):
                    self._progress = progress_obj
                    self._task = task
                
                def advance(self, amount: int = 1):
                    """
                    [KOR] 프로그레스 증가
                    [ENG] Increment progress
                    
                    Args:
                        amount: [KOR] 증가량 / [ENG] Amount to advance
                    """
                    self._progress.update(self._task, advance=amount)
                
                def update_description(self, description: str):
                    """
                    [KOR] 프로그레스 설명 업데이트
                    [ENG] Update progress description
                    
                    Args:
                        description: [KOR] 새 설명 / [ENG] New description
                    """
                    self._progress.update(self._task, description=description)
            
            yield ProgressContext(progress, task_id)


# [KOR] 기본 로거 인스턴스 (싱글톤 패턴)
# [ENG] Default logger instance (singleton pattern)
_default_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """
    [KOR] 기본 로거 인스턴스 반환
    [ENG] Get default logger instance
    
    Returns:
        Logger: [KOR] 기본 로거 인스턴스 / [ENG] Default logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = Logger()
    return _default_logger
