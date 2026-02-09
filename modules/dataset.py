# [KOR] 비디오 데이터셋 로딩 및 관리 모듈
# [ENG] Video dataset loading and management module

import os
import glob
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """
    [KOR] 비디오 정보를 담는 데이터 클래스
    [ENG] Data class to hold video information
    
    Attributes:
        path: [KOR] 비디오 파일 경로 / [ENG] Path to video file
        name: [KOR] 비디오 이름 (확장자 제외) / [ENG] Video name (without extension)
        working_dir: [KOR] 작업 디렉토리 경로 / [ENG] Path to working directory
        tubes_dir: [KOR] 튜브 디렉토리 경로 / [ENG] Path to tubes directory
    """
    path: str
    name: str
    working_dir: str
    tubes_dir: str


class VideoDataset:
    """
    [KOR] 비디오 데이터셋 클래스
    [ENG] Video dataset class
    
    [KOR] 입력 디렉토리에서 비디오 파일을 검색하고 관리합니다.
    [ENG] Searches and manages video files from input directory.
    """
    
    # [KOR] 지원하는 비디오 확장자
    # [ENG] Supported video extensions
    SUPPORTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    def __init__(self, 
                 dataset: str = "demo",
                 input_dir: str = "./data/input",
                 working_dir: str = "./data/working",
                 output_dir: str = "./data/output"):
        """
        [KOR] VideoDataset 초기화
        [ENG] Initialize VideoDataset
        
        Args:
            input_dir: [KOR] 입력 비디오 디렉토리 / [ENG] Input video directory
            working_dir: [KOR] 작업 디렉토리 / [ENG] Working directory
            output_dir: [KOR] 출력 디렉토리 / [ENG] Output directory
        """
        self.input_dir = os.path.join(input_dir, dataset)
        self.working_dir = os.path.join(working_dir, dataset)
        self.output_dir = os.path.join(output_dir, dataset)
        
        # [KOR] 디렉토리 생성
        # [ENG] Create directories
        self._setup_directories()
        
        # [KOR] 비디오 목록
        # [ENG] Video list
        self._videos: List[VideoInfo] = []
        self._loaded = False
    
    def _setup_directories(self):
        """
        [KOR] 필요한 디렉토리 생성
        [ENG] Create necessary directories
        """
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load(self, extensions: Optional[List[str]] = None) -> int:
        """
        [KOR] 입력 디렉토리에서 비디오 파일 로드
        [ENG] Load video files from input directory
        
        Args:
            extensions: [KOR] 검색할 확장자 목록 (기본값: 지원 확장자 전체)
                       [ENG] List of extensions to search (default: all supported)
        
        Returns:
            [KOR] 발견된 비디오 수
            [ENG] Number of videos found
        """
        if extensions is None:
            extensions = self.SUPPORTED_EXTENSIONS
        
        self._videos = []
        
        # [KOR] 각 확장자에 대해 재귀적으로 검색
        # [ENG] Search recursively for each extension
        for ext in extensions:
            # [KOR] 대소문자 모두 검색
            # [ENG] Search both cases
            pattern_lower = os.path.join(self.input_dir, "**", f"*{ext.lower()}")
            pattern_upper = os.path.join(self.input_dir, "**", f"*{ext.upper()}")
            
            for pattern in [pattern_lower, pattern_upper]:
                for video_path in glob.glob(pattern, recursive=True):
                    # [KOR] 중복 방지
                    # [ENG] Prevent duplicates
                    if any(v.path == video_path for v in self._videos):
                        continue
                    
                    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
                    cur_working_dir = os.path.join(self.working_dir, video_name)
                    tubes_dir = os.path.join(cur_working_dir, "tubes")
                    
                    self._videos.append(VideoInfo(
                        path=video_path,
                        name=video_name,
                        working_dir=cur_working_dir,
                        tubes_dir=tubes_dir
                    ))
        
        # [KOR] 이름순 정렬
        # [ENG] Sort by name
        self._videos.sort(key=lambda v: v.name)
        self._loaded = True
        
        return len(self._videos)
    
    def __len__(self) -> int:
        """
        [KOR] 비디오 수 반환
        [ENG] Return number of videos
        """
        return len(self._videos)
    
    def __iter__(self):
        """
        [KOR] 비디오 순회 이터레이터
        [ENG] Iterator for videos
        """
        return iter(self._videos)
    
    def __getitem__(self, index: int) -> VideoInfo:
        """
        [KOR] 인덱스로 비디오 정보 접근
        [ENG] Access video info by index
        
        Args:
            index: [KOR] 비디오 인덱스 / [ENG] Video index
            
        Returns:
            [KOR] 비디오 정보 / [ENG] Video information
        """
        return self._videos[index]
    
    @property
    def videos(self) -> List[VideoInfo]:
        """
        [KOR] 모든 비디오 목록 반환
        [ENG] Return all videos list
        """
        return self._videos
    
    @property
    def is_empty(self) -> bool:
        """
        [KOR] 비디오가 없는지 확인
        [ENG] Check if no videos found
        """
        return len(self._videos) == 0
    
    def get_video_paths(self) -> List[str]:
        """
        [KOR] 모든 비디오 경로 목록 반환
        [ENG] Return list of all video paths
        
        Returns:
            [KOR] 비디오 경로 리스트
            [ENG] List of video paths
        """
        return [v.path for v in self._videos]
    
    def get_video_names(self) -> List[str]:
        """
        [KOR] 모든 비디오 이름 목록 반환
        [ENG] Return list of all video names
        
        Returns:
            [KOR] 비디오 이름 리스트
            [ENG] List of video names
        """
        return [v.name for v in self._videos]


def load_videos(input_dir: str = "./data/input",
                working_dir: str = "./data/working",
                output_dir: str = "./data/output",
                extensions: Optional[List[str]] = None) -> VideoDataset:
    """
    [KOR] 비디오 데이터셋을 로드하는 편의 함수
    [ENG] Convenience function to load video dataset
    
    Args:
        input_dir: [KOR] 입력 비디오 디렉토리 / [ENG] Input video directory
        working_dir: [KOR] 작업 디렉토리 / [ENG] Working directory
        output_dir: [KOR] 출력 디렉토리 / [ENG] Output directory
        extensions: [KOR] 검색할 확장자 목록 / [ENG] List of extensions to search
    
    Returns:
        [KOR] 로드된 VideoDataset
        [ENG] Loaded VideoDataset
    """
    dataset = VideoDataset(input_dir, working_dir, output_dir)
    dataset.load(extensions)
    return dataset
