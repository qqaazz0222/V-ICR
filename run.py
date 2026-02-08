# [KOR] V-ICR 메인 실행 스크립트
# [ENG] V-ICR Main Execution Script

import os
import time
import argparse
from modules.detector import Detector
from modules.recognizer import Recognizer
from modules.exporter import export_action_labels
from modules.dataset import VideoDataset
from utils.logger import Logger, format_time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def parse_args():
    """
    [KOR] 명령줄 인자를 파싱합니다.
    [ENG] Parse command line arguments.
    
    Returns:
        [KOR] 파싱된 인자 객체
        [ENG] Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="V-ICR Video Processor")
    parser.add_argument("--skip-recognition", action="store_true",
                        help="Skip the recognition phase (detection only)")
    parser.add_argument("--refinement-iterations", type=int, default=5,
                        help="Number of refinement iterations for recognition (default: 5)")
    parser.add_argument("--input-dir", type=str, default="./data/input",
                        help="Input video directory (default: ./data/input)")
    parser.add_argument("--working-dir", type=str, default="./data/working",
                        help="Working directory (default: ./data/working)")
    parser.add_argument("--output-dir", type=str, default="./data/output",
                        help="Output directory (default: ./data/output)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger = Logger()
    start_total = time.time()
    
    # [KOR] 헤더 출력
    # [ENG] Print header
    logger.print_header("V-ICR Video Processor")
    
    # [KOR] 비디오 데이터셋 로드
    # [ENG] Load video dataset
    dataset = VideoDataset(
        input_dir=args.input_dir,
        working_dir=args.working_dir,
        output_dir=args.output_dir
    )
    num_videos = dataset.load()
    
    if dataset.is_empty:
        logger.print_warning("No video files found in input directory.")
        exit(0)
    
    logger.print_info(f"Found {num_videos} video(s)")
    
    # [KOR] 사전 정의된 label_map.txt 로드
    # [ENG] Load predefined label_map.txt if exists
    predefined_labelmap = None
    labelmap_path = os.path.join(args.input_dir, "label_map.txt")
    if os.path.exists(labelmap_path):
        with open(labelmap_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # [KOR] 빈 줄과 주석 제거, 라벨만 추출
        # [ENG] Remove empty lines and comments, extract labels only
        labels = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # [KOR] "0: label" 형식 또는 "label" 형식 지원
                # [ENG] Support both "0: label" and "label" formats
                if ":" in line:
                    label = line.split(":", 1)[1].strip()
                else:
                    label = line
                if label:
                    labels.append(label)
        
        if labels:
            predefined_labelmap = labels
            logger.print_info(f"Loaded predefined label-map: {len(labels)} labels")
    
    # [KOR] Detector 초기화 (YOLO12 + ByteTrack)
    # [ENG] Initialize Detector (YOLO12 + ByteTrack)
    with logger.status("Initializing Detector..."):
        detector = Detector()
    logger.print_success("Detector initialized")
    
    # [KOR] Recognizer 초기화 (필요한 경우)
    # [ENG] Initialize Recognizer (if needed)
    recognizer = None
    if not args.skip_recognition:
        with logger.status("Initializing Recognizer (Qwen3-VL)..."):
            recognizer = Recognizer()
        logger.print_success("Recognizer initialized")
    
    logger.print("")
    
    # [KOR] 처리 결과 저장 리스트
    # [ENG] List to store processing results
    results = []
    
    # [KOR] 프로그레스 바와 함께 처리
    # [ENG] Process with progress bar
    with logger.progress_bar(len(dataset), "Processing") as progress_ctx:
        for video in dataset:
            progress_ctx.update_description(f"Processing: {video.name}")
            
            start_time = time.time()
            
            try:
                # [KOR] Phase 1: 탐지 및 추적
                # [ENG] Phase 1: Detection and Tracking
                detector.infer(video.path, video.working_dir)
                
                # [KOR] Phase 2: 행동 인식 (활성화된 경우)
                # [ENG] Phase 2: Action Recognition (if enabled)
                if recognizer is not None:
                    recognizer.recognize(
                        video.path, 
                        video.tubes_dir,
                        iterations=args.refinement_iterations,
                        predefined_labelmap=predefined_labelmap
                    )
                    
                    # [KOR] Phase 3: 최종 라벨 데이터를 output 디렉토리에 저장
                    # [ENG] Phase 3: Export final labels to output directory
                    print("    [>] Exporting action labels...")
                    export_data = export_action_labels(
                        video.path, 
                        video.working_dir, 
                        dataset.output_dir
                    )
                    print(f"        Exported: {video.name}.json ({export_data['num_persons']} persons)")
                
                elapsed = time.time() - start_time
                results.append({
                    "name": video.name,
                    "status": "success",
                    "time": elapsed
                })
                logger.print_item_result(video.name, success=True, elapsed_time=elapsed)
                
            except Exception as e:
                # [KOR] 예외 발생 시 실패 기록
                # [ENG] Record failure on exception
                elapsed = time.time() - start_time
                results.append({
                    "name": video.name,
                    "status": "failed",
                    "time": elapsed,
                    "error": str(e)
                })
                logger.print_item_result(video.name, success=False, elapsed_time=elapsed, error=str(e))
            
            progress_ctx.advance()
    
    total_time = time.time() - start_total
    
    # [KOR] 처리 요약 출력
    # [ENG] Print processing summary
    logger.print_summary(results, total_time)