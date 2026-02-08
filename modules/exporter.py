# [KOR] 최종 행동 라벨 데이터를 output 디렉토리로 내보내기
# [ENG] Export final action labels to output directory

import os
import json
import cv2
from typing import Dict, Any, List


def export_action_labels(video_path: str, working_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    [KOR] metadata와 recognition_results를 조합하여 행동 라벨을 내보냅니다.
    [ENG] Export action labels by combining metadata and recognition results.
    
    [KOR] 다음을 포함하는 종합 라벨 파일을 생성합니다:
    [ENG] Creates a comprehensive action label file that includes:
    - [KOR] 비디오 메타데이터 (fps, 해상도, 길이)
    - [ENG] Video metadata (fps, resolution, duration)
    - [KOR] 바운딩 박스가 포함된 사람 트랙
    - [ENG] Person tracks with bounding boxes
    - [KOR] 프레임별 행동 라벨
    - [ENG] Per-frame action labels
    - [KOR] 행동 카테고리 Labelmap
    - [ENG] Labelmap for action categories
    
    Args:
        video_path: [KOR] 원본 비디오 경로 / [ENG] Path to original video
        working_dir: [KOR] 작업 디렉토리 / [ENG] Working directory with tubes/metadata.json
        output_dir: [KOR] 출력 디렉토리 / [ENG] Output directory to save the label file
        
    Returns:
        [KOR] 내보낸 라벨 데이터 딕셔너리
        [ENG] Dictionary with exported label data
    """
    tubes_dir = os.path.join(working_dir, "tubes")
    metadata_path = os.path.join(tubes_dir, "metadata.json")
    recognition_path = os.path.join(tubes_dir, "recognition_results.json")
    labelmap_path = os.path.join(working_dir, "label_map.txt")
    
    # [KOR] 데이터 로드
    # [ENG] Load data
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    with open(recognition_path, "r") as f:
        recognition = json.load(f)
    
    # [KOR] 비디오 정보 추출
    # [ENG] Get video info
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    video_duration = total_frames / video_fps if video_fps > 0 else 0
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    
    # [KOR] Labelmap 구성 (행동 ID -> 행동 이름 매핑)
    # [ENG] Build labelmap (action ID -> action name mapping)
    labelmap = {}
    action_to_id = {}
    for i, action in enumerate(recognition.get("labelmap", [])):
        labelmap[i] = {
            "name": action,
            "original_actions": recognition.get("action_groups", {}).get(action, [action])
        }
        action_to_id[action] = i
    
    # [KOR] 프레임 레벨 주석이 포함된 사람 트랙 구성
    # [ENG] Build person tracks with frame-level annotations
    persons = {}
    
    for person_id, meta in metadata.items():
        # [KOR] 이 사람의 인식 데이터 가져오기
        # [ENG] Get recognition data for this person
        tube_data = recognition.get("tubes", {}).get(person_id, {})
        final_actions = tube_data.get("final_actions", [])
        temporal_labels = tube_data.get("temporal_labels", {})
        
        # [KOR] 행동 타임라인 생성 (초 -> 행동)
        # [ENG] Create action timeline (second -> action)
        # [KOR] 최고 확률 행동만 저장 (action, action_id)
        # [ENG] Store only top action (action, action_id)
        action_timeline = {}
        for action_entry in final_actions:
            sec = action_entry["time"]
            action_timeline[sec] = {
                "action": action_entry["action"],
                "action_id": action_to_id.get(action_entry["action"], -1)
            }
        
        # [KOR] 프레임 레벨 주석 생성
        # [ENG] Create frame-level annotations
        frames = []
        for bbox_entry in meta.get("bboxes", []):
            frame_idx = bbox_entry["frame_idx"]
            box = bbox_entry["box"]  # [x1, y1, x2, y2]
            
            # [KOR] 이 프레임이 속하는 초 결정
            # [ENG] Determine which second this frame belongs to
            second = int(frame_idx / video_fps) if video_fps > 0 else 0
            
            # [KOR] 해당 초의 행동 가져오기
            # [ENG] Get action for this second
            action_info = action_timeline.get(second, {
                "action": "unknown",
                "action_id": -1
            })
            
            frames.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / video_fps if video_fps > 0 else 0,
                "bbox": {
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3]
                },
                "action": action_info["action"],
                "action_id": action_info["action_id"]
            })
        
        # [KOR] 사람 데이터 구성
        # [ENG] Build person data
        persons[person_id] = {
            "id": person_id,
            "start_frame": meta.get("start_frame", 0),
            "end_frame": meta.get("end_frame", 0),
            "start_time": meta.get("start_frame", 0) / video_fps if video_fps > 0 else 0,
            "end_time": meta.get("end_frame", 0) / video_fps if video_fps > 0 else 0,
            "tube_size": {
                "width": meta.get("width", 0),
                "height": meta.get("height", 0)
            },
            "action_summary": list(set(a["action"] for a in final_actions)),
            "action_timeline": action_timeline,
            "frames": frames
        }
    
    # [KOR] 최종 출력 구조 생성
    # [ENG] Build final output structure
    output_data = {
        "version": "1.0",
        "video": {
            "name": video_name,
            "path": video_path,
            "fps": video_fps,
            "width": video_width,
            "height": video_height,
            "total_frames": total_frames,
            "duration": video_duration
        },
        "labelmap": labelmap,
        "num_persons": len(persons),
        "persons": persons,
        "summary": {
            "total_action_instances": sum(
                len(p.get("action_timeline", {})) 
                for p in persons.values()
            ),
            "action_distribution": {}
        }
    }
    
    # [KOR] 행동 분포 계산
    # [ENG] Calculate action distribution
    action_counts = {}
    for person in persons.values():
        for sec, action_info in person.get("action_timeline", {}).items():
            action = action_info["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
    output_data["summary"]["action_distribution"] = action_counts
    
    # [KOR] output 디렉토리에 저장
    # [ENG] Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_name}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_data


def main():
    """
    [KOR] 내보내기 함수 테스트
    [ENG] Test export function
    """
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python exporter.py <video_path> <working_dir> <output_dir>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    working_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    result = export_action_labels(video_path, working_dir, output_dir)
    
    print(f"Exported to: {output_dir}/{result['video']['name']}.json")
    print(f"Persons: {result['num_persons']}")
    print(f"Actions: {result['summary']['action_distribution']}")


if __name__ == "__main__":
    main()
