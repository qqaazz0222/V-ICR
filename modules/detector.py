# [KOR] YOLO와 ByteTrack을 사용한 사람 탐지 및 추적
# [ENG] Person detection and tracking using YOLO and ByteTrack

import os
import cv2
import shutil
import numpy as np
from ultralytics import YOLO


class Detector:
    """
    [KOR] YOLO12와 ByteTrack을 사용한 사람 탐지 및 추적 클래스
    [ENG] Person detection and tracking class using YOLO12 and ByteTrack
    
    [KOR] 이 클래스는 비디오에서 사람을 탐지하고 추적하여 개인별 튜브를 추출합니다.
    [ENG] This class detects and tracks persons in video and extracts individual tubes.
    
    Features:
    - [KOR] YOLO12 기반 사람 탐지 / [ENG] YOLO12-based person detection
    - [KOR] ByteTrack 멀티오브젝트 추적 / [ENG] ByteTrack multi-object tracking
    - [KOR] Kalman 필터 스무딩 / [ENG] Kalman filter smoothing
    - [KOR] 끊어진 트랙 스티칭 / [ENG] Broken track stitching
    """
    
    def __init__(self, model_path="./checkpoints/yolo12x.pt", min_tube_length=30):
        """
        [KOR] Detector 초기화
        [ENG] Initialize Detector
        
        Args:
            model_path: [KOR] YOLO 모델 가중치 경로 / [ENG] Path to YOLO model weights
            min_tube_length: [KOR] 최소 튜브 길이 (프레임 수) / [ENG] Minimum tube length (in frames)
        """
        # [KOR] 모델 로딩은 호출자가 로그 기록
        # [ENG] Model loading is silent - logged by caller
        self.model = YOLO(model_path)
        self.track_results = None
        self.min_tube_length = min_tube_length
    
    def _detect(self, video_path, save_dir):
        """
        [KOR] 시각화/검증을 위한 탐지 실행
        [ENG] Run detection for visualization/verification
        
        Args:
            video_path: [KOR] 입력 비디오 경로 / [ENG] Path to input video
            save_dir: [KOR] 결과 저장 디렉토리 / [ENG] Directory to save results
        """
        abs_save_dir = os.path.abspath(save_dir)
        self.model.predict(
            source=video_path,
            save=True,
            project=abs_save_dir,
            name='detect',
            exist_ok=True,
            classes=[0],  # [KOR] Person만 / [ENG] Person only
            verbose=False,
            stream=True
        )

    def _track(self, video_path, save_dir):
        """
        [KOR] 추적을 실행하고 결과를 메모리에 저장
        [ENG] Run tracking and store results in memory
        
        Args:
            video_path: [KOR] 입력 비디오 경로 / [ENG] Path to input video
            save_dir: [KOR] 결과 저장 디렉토리 / [ENG] Directory to save results
        """
        abs_save_dir = os.path.abspath(save_dir)
        self.track_results = self.model.track(
            source=video_path,
            tracker="modules/bytetrack_tuned.yaml",
            persist=True,
            classes=[0],  # [KOR] Person만 / [ENG] Person only
            conf=0.6,
            save=False,
            project=abs_save_dir,
            name='track',
            exist_ok=True,
            verbose=False,  # [KOR] 노이즈 감소 / [ENG] Reduce noise
            stream=True
        )

    def _extract_tubes(self, video_path, save_dir):
        """
        [KOR] 추적 결과에서 개인별 튜브 추출
        [ENG] Extract individual tubes from tracking results
        
        Args:
            video_path: [KOR] 입력 비디오 경로 / [ENG] Path to input video
            save_dir: [KOR] 결과 저장 디렉토리 / [ENG] Directory to save results
        """
        import json
        from collections import defaultdict
        
        if self.track_results is None:
            self._track(video_path, save_dir)
            
        print("    [>] Extracting tubes...")
        
        output_dir = os.path.join(save_dir, "tubes")
        os.makedirs(output_dir, exist_ok=True)
        
        # ==========================================
        # [KOR] Pass 1: 추적 결과 파싱 & 통계 계산
        # [ENG] Pass 1: Parse Tracking Results & Compute Stats
        # ==========================================
        
        # [KOR] track_id -> 통계 딕셔너리
        # [ENG] track_id -> stats dictionary
        track_stats = defaultdict(lambda: {
            'max_w': 0, 'max_h': 0, 
            'count': 0, 
            'start_frame': float('inf'), 
            'end_frame': float('-inf')
        })
        
        # [KOR] Pass 2를 위해 프레임별 탐지 저장 (track 재실행 방지)
        # [ENG] Use a list to store frame-by-frame detections for Pass 2 to avoid re-running tracks
        video_detections = []
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # [KOR] 생성기/리스트 순회
        # [ENG] Iterate generator/list
        for frame_idx, result in enumerate(self.track_results):
            frame_dets = {}
            
            boxes = result.boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()
                
                for track_id, box in zip(track_ids, xyxys):
                    x1, y1, x2, y2 = map(int, box)
                    w = x2 - x1
                    h = y2 - y1
                    
                    # [KOR] 통계 업데이트
                    # [ENG] Update stats
                    stats = track_stats[track_id]
                    stats['max_w'] = max(stats['max_w'], w)
                    stats['max_h'] = max(stats['max_h'], h)
                    stats['count'] += 1
                    stats['start_frame'] = min(stats['start_frame'], frame_idx)
                    stats['end_frame'] = max(stats['end_frame'], frame_idx)
                    
                    frame_dets[track_id] = (x1, y1, x2, y2)
            
            video_detections.append(frame_dets)

        # ==========================================
        # [KOR] 후처리: 끊어진 트랙 스티칭
        # [ENG] Post-Process: Stitch Broken Tracks
        # ==========================================

        # [KOR] 1. 스티칭을 위한 종합 통계 수집
        # [ENG] 1. Gather comprehensive stats for stitching
        track_lifecycle = defaultdict(lambda: {
            'start_frame': float('inf'), 'end_frame': float('-inf'),
            'first_box': None, 'last_box': None
        })

        for frame_idx, dets in enumerate(video_detections):
            for tid, box in dets.items():
                tl = track_lifecycle[tid]
                if frame_idx < tl['start_frame']:
                    tl['start_frame'] = frame_idx
                    tl['first_box'] = box
                if frame_idx > tl['end_frame']:
                    tl['end_frame'] = frame_idx
                    tl['last_box'] = box
        
        # [KOR] 2. 병합 로직
        # [ENG] 2. Logic to merge
        sorted_tids = sorted(track_lifecycle.keys(), key=lambda x: track_lifecycle[x]['start_frame'])
        
        id_map = {tid: tid for tid in sorted_tids}
        
        # [KOR] 사용자 정의 파라미터: n=3초
        # [ENG] User defined params: n=3 seconds
        max_gap_seconds = 3.0 
        max_gap_frames = max_gap_seconds * fps
        dist_thresh = 100.0  # [KOR] 픽셀 거리 임계값 / [ENG] Pixel distance threshold
        
        unified_state = {}  # [KOR] master_id -> 상태 / [ENG] master_id -> state
        
        for tid in sorted_tids:
            # [KOR] 현재 트랙 정보
            # [ENG] Current track info
            curr_info = track_lifecycle[tid]
            curr_start = curr_info['start_frame']
            curr_box = curr_info['first_box']
            curr_cx, curr_cy = (curr_box[0]+curr_box[2])/2, (curr_box[1]+curr_box[3])/2
            
            best_match_master = None
            min_dist = float('inf')
            
            # [KOR] 선행 트랙 찾기
            # [ENG] Find a predecessor
            for master_id, state in unified_state.items():
                # [KOR] 시간 갭 확인 (갭은 양수이고 임계값 이하여야 함)
                # [ENG] Check time gap (gap must be positive but < threshold)
                gap = curr_start - state['last_frame']
                
                if 0 < gap <= max_gap_frames:
                    # [KOR] 공간 거리 확인
                    # [ENG] Check spatial distance
                    prev_box = state['last_box']
                    prev_cx, prev_cy = (prev_box[0]+prev_box[2])/2, (prev_box[1]+prev_box[3])/2
                    
                    dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                    
                    if dist < dist_thresh:
                        if dist < min_dist:
                            min_dist = dist
                            best_match_master = master_id
            
            if best_match_master is not None:
                # [KOR] 병합!
                # [ENG] Merge!
                id_map[tid] = best_match_master
                
                # [KOR] 갭에 대한 선형 보간
                # [ENG] Linear Interpolation for Gap
                prev_end = unified_state[best_match_master]['last_frame']
                curr_start = curr_info['start_frame']
                box_start = np.array(unified_state[best_match_master]['last_box'])
                box_end = np.array(curr_info['first_box'])
                
                # [KOR] 갭 프레임 채우기
                # [ENG] Fill gap frames
                for f_idx in range(prev_end + 1, curr_start):
                    alpha = (f_idx - prev_end) / (curr_start - prev_end)
                    interp_box = box_start * (1 - alpha) + box_end * alpha
                    # [KOR] 정수 튜플로 변환
                    # [ENG] Convert to int tuple
                    int_box = tuple(map(int, interp_box))
                    
                    # [KOR] MASTER ID에 대해 video_detections에 추가
                    # [ENG] Add to video_detections for the MASTER ID
                    if f_idx < len(video_detections):
                        video_detections[f_idx][best_match_master] = int_box

                # [KOR] 마스터의 상태 업데이트 - 현재 것으로 프레임과 박스 업데이트 (확장)
                # [ENG] Update the master's state - Update frame and box to current one (extending it)
                unified_state[best_match_master]['last_frame'] = curr_info['end_frame']
                unified_state[best_match_master]['last_box'] = curr_info['last_box']
            else:
                # [KOR] 새 마스터
                # [ENG] New master
                unified_state[tid] = {
                    'last_frame': curr_info['end_frame'],
                    'last_box': curr_info['last_box']
                }

        # ==========================================
        # [KOR] 3. ID 맵 적용 및 Kalman 스무딩 수행
        # [ENG] 3. Apply ID Map and Perform Kalman Smoothing
        # ==========================================
        
        # [KOR] 3.1 Master ID별로 관측치 그룹화
        # [ENG] 3.1 Group observations by Master ID
        master_trajectories = defaultdict(list)  # master_tid -> list of (frame_idx, box)
        
        for frame_idx, dets in enumerate(video_detections):
            for tid, box in dets.items():
                if tid in id_map:
                    master_tid = id_map[tid]
                else:
                    master_tid = tid  # [KOR] 보간으로 추가된 경우 자기 자신이 마스터 / [ENG] Should be master itself if added by interpolation
                
                master_trajectories[master_tid].append((frame_idx, box))
                
        # [KOR] 3.2 Kalman 스무더 정의
        # [ENG] 3.2 Define Kalman Smoother
        def smooth_track_kf(measurements):
            """
            [KOR] Kalman 필터를 사용한 트랙 스무딩
            [ENG] Smooth track using Kalman filter
            
            Args:
                measurements: [KOR] (프레임, (x1,y1,x2,y2)) 리스트 / [ENG] list of (frame, (x1,y1,x2,y2)) sorted by frame
                
            Returns:
                [KOR] 스무딩된 측정치 리스트 / [ENG] List of smoothed measurements
            """
            if not measurements: 
                return []
            measurements.sort(key=lambda x: x[0])
            
            # [KOR] Kalman 필터 초기화
            # [ENG] Init Kalman Filter
            # [KOR] 상태: [cx, cy, w, h, vx, vy, vw, vh]
            # [ENG] State: [cx, cy, w, h, vx, vy, vw, vh]
            kf = cv2.KalmanFilter(8, 4)
            kf.measurementMatrix = np.array([
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0]
            ], np.float32)
            
            kf.transitionMatrix = np.array([
                [1,0,0,0,1,0,0,0],
                [0,1,0,0,0,1,0,0],
                [0,0,1,0,0,0,1,0],
                [0,0,0,1,0,0,0,1],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1]
            ], np.float32)
            
            # [KOR] 프로세스 노이즈 (스무딩 vs 반응성)
            # [ENG] Process Noise (Smoothness vs Reaction)
            # [KOR] 낮음 = 더 스무딩, 높음 = 측정값을 더 빨리 따라감
            # [ENG] Lower = smoother, Higher = follows measurement faster
            # [KOR] 위치(cx, cy, w, h)에는 매우 낮은 노이즈, 속도에는 약간 더 높은 노이즈
            # [ENG] Very low noise for position (cx, cy, w, h), slightly higher for velocity
            process_noise = np.diag([
                0.0005,  # cx - [KOR] 매우 작은 위치 변화만 허용 / [ENG] Allow only very small position change
                0.0005,  # cy
                0.0001,  # w - [KOR] 크기는 거의 변화 없음 / [ENG] Size almost no change
                0.0001,  # h
                0.001,   # vx - [KOR] 속도의 변화는 약간 허용 / [ENG] Slightly allow velocity change
                0.001,   # vy
                0.0005,  # vw
                0.0005   # vh
            ]).astype(np.float32)
            kf.processNoiseCov = process_noise
            
            # [KOR] 측정 노이즈 (탐지에 대한 신뢰도)
            # [ENG] Measurement Noise (Trust in detection)
            # [KOR] 높음 = 모델 신뢰 (더 스무딩), 낮음 = 측정 신뢰
            # [ENG] Higher = trust model (smoother), Lower = trust measurement
            # [KOR] 위치보다 크기의 측정 노이즈를 더 높게 설정 (크기가 더 안정적이도록)
            # [ENG] Set higher measurement noise for size than position (for more stable size)
            measurement_noise = np.diag([
                100.0,   # cx - [KOR] 위치 측정을 덜 신뢰 (더 스무딩) / [ENG] Less trust in position measurement (more smoothing)
                100.0,   # cy
                200.0,   # w - [KOR] 크기 측정을 훨씬 덜 신뢰 / [ENG] Much less trust in size measurement
                200.0    # h
            ]).astype(np.float32)
            kf.measurementNoiseCov = measurement_noise
            
            # [KOR] 첫 측정으로 상태 초기화
            # [ENG] Init state with first measurement
            f0, b0 = measurements[0]
            x1, y1, x2, y2 = b0
            w, h = x2-x1, y2-y1
            cx, cy = x1 + w/2, y1 + h/2
            
            kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float32)
            kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.01  # [KOR] 더 작은 초기 공분산 / [ENG] Smaller initial covariance
            
            smoothed = []
            framemap = {f: b for f, b in measurements}
            min_f, max_f = measurements[0][0], measurements[-1][0]
            
            for f in range(min_f, max_f + 1):
                # [KOR] 예측
                # [ENG] Predict
                pred = kf.predict()
                
                # [KOR] 보정
                # [ENG] Correct
                if f in framemap:
                    m_box = framemap[f]
                    m_x1, m_y1, m_x2, m_y2 = m_box
                    m_w, m_h = m_x2-m_x1, m_y2-m_y1
                    m_cx, m_cy = m_x1 + m_w/2, m_y1 + m_h/2
                    
                    z = np.array([m_cx, m_cy, m_w, m_h], np.float32)
                    kf.correct(z)
                    s = kf.statePost
                else:
                    s = pred
                    
                sx, sy, sw, sh = s[0], s[1], s[2], s[3]
                
                # [KOR] xyxy로 다시 변환
                # [ENG] Convert back to xyxy
                nx1 = int(sx - sw/2)
                ny1 = int(sy - sh/2)
                nx2 = int(sx + sw/2)
                ny2 = int(sy + sh/2)
                
                smoothed.append((f, (nx1, ny1, nx2, ny2)))
                
            return smoothed
            
        # [KOR] 3.3 스무딩 적용 및 video_detections 재구성
        # [ENG] 3.3 Apply Smoothing and Re-populate video_detections
        new_track_stats = defaultdict(lambda: {
            'max_w': 0, 'max_h': 0, 
            'count': 0, 
            'start_frame': float('inf'), 
            'end_frame': float('-inf')
        })
        
        # [KOR] 스무딩된 궤적으로 재구성하기 위해 이전 탐지 삭제
        # [ENG] Clear old detections to rebuild from smoothed trajectories
        max_frame_idx = len(video_detections)
        new_video_detections = [{} for _ in range(max_frame_idx)]
        
        for tid, measurements in master_trajectories.items():
            if len(measurements) < 5:
                # [KOR] 신뢰할 수 있는 KF에는 너무 짧음, 원본 유지
                # [ENG] Too short for reliable KF, just keep original
                smoothed = measurements
            else:
                smoothed = smooth_track_kf(measurements)
            
            for f, box in smoothed:
                if f < max_frame_idx:
                    new_video_detections[f][tid] = box
                    
                    # [KOR] 통계 업데이트
                    # [ENG] Update stats
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    stats = new_track_stats[tid]
                    stats['max_w'] = max(stats['max_w'], w)
                    stats['max_h'] = max(stats['max_h'], h)
                    stats['count'] += 1
                    stats['start_frame'] = min(stats['start_frame'], f)
                    stats['end_frame'] = max(stats['end_frame'], f)
                    
        video_detections = new_video_detections
        track_stats = new_track_stats
            
        # ==========================================
        # [KOR] 트랙 필터링 & Writer 준비
        # [ENG] Filter Tracks & Prepare Writers
        # ==========================================
        valid_tracks = {}
        tube_writers = {}
        
        for tid, stats in track_stats.items():
            if stats['count'] >= self.min_tube_length:
                valid_tracks[tid] = stats
                
                # [KOR] MAX 크기로 writer 준비
                # [ENG] Prepare writer with MAX size
                tube_path = os.path.join(output_dir, f"id_{tid}.mp4")
                target_w = stats['max_w']
                target_h = stats['max_h']
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(tube_path, fourcc, fps, (target_w, target_h))
                
                tube_writers[tid] = {
                    'writer': writer,
                    'width': target_w,
                    'height': target_h
                }
        
        print(f"        Tracks: {len(valid_tracks)} valid / {len(track_stats)} total")
        
        # ==========================================
        # [KOR] Pass 2: 프레임 추출 & 저장
        # [ENG] Pass 2: Extract & Write Frames
        # ==========================================
        
        # [KOR] 읽기를 위해 비디오 다시 열기
        # [ENG] Re-open video for reading
        cap.release()
        cap = cv2.VideoCapture(video_path)
        
        def adjust_bbox_to_aspect(box, target_aspect, img_w, img_h):
            """
            [KOR] 바운딩 박스를 목표 종횡비에 맞게 조정
            [ENG] Adjust bounding box to target aspect ratio
            
            Args:
                box: [KOR] (x1, y1, x2, y2) 박스 / [ENG] (x1, y1, x2, y2) box
                target_aspect: [KOR] 목표 종횡비 / [ENG] Target aspect ratio
                img_w: [KOR] 이미지 너비 / [ENG] Image width
                img_h: [KOR] 이미지 높이 / [ENG] Image height
                
            Returns:
                [KOR] 조정된 박스 / [ENG] Adjusted box
            """
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if h <= 0 or w <= 0: 
                return box
            
            current_aspect = w / h
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            if current_aspect < target_aspect:
                # [KOR] 너무 세로로 길음, 너비 증가
                # [ENG] Too tall, increase width
                new_w = h * target_aspect
                new_h = h
            else:
                # [KOR] 너무 가로로 길음, 높이 증가
                # [ENG] Too wide, increase height
                new_h = w / target_aspect
                new_w = w
                
            nx1 = int(cx - new_w / 2)
            nx2 = int(cx + new_w / 2)
            ny1 = int(cy - new_h / 2)
            ny2 = int(cy + new_h / 2)
            
            # [KOR] 클램핑 로직 (부분적으로 범위 밖 허용, letterbox가 처리)
            # [ENG] Clamp logic (relaxed to allow partial out of bounds, letterbox handles it)
            nx1 = max(0, nx1)
            ny1 = max(0, ny1)
            nx2 = min(img_w, nx2)
            ny2 = min(img_h, ny2)
            
            return (nx1, ny1, nx2, ny2)

        def letterbox_resize(img, target_size):
            """
            [KOR] 레터박스 방식으로 리사이즈 (검은색 패딩)
            [ENG] Letterbox resize (with black padding)
            
            Args:
                img: [KOR] 입력 이미지 / [ENG] Input image
                target_size: [KOR] (width, height) 목표 크기 / [ENG] (width, height) target size
                
            Returns:
                [KOR] 리사이즈된 이미지 / [ENG] Resized image
            """
            target_w, target_h = target_size
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                return np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
            scale = min(target_w / w, target_h / h)
            nw, nh = int(w * scale), int(h * scale)
            
            resized = cv2.resize(img, (nw, nh))
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_off = (target_w - nw) // 2
            y_off = (target_h - nh) // 2
            canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
            return canvas

        for frame_idx, frame_dets in enumerate(video_detections):
            ret, frame = cap.read()
            if not ret:
                break
            
            for tid, box in frame_dets.items():
                if tid in valid_tracks:
                    tw = tube_writers[tid]
                    target_w, target_h = tw['width'], tw['height']
                    target_aspect = target_w / target_h if target_h > 0 else 1.0
                    
                    # [KOR] 1. 종횡비 조정
                    # [ENG] 1. Aspect Adjust
                    adj_box = adjust_bbox_to_aspect(box, target_aspect, width, height)
                    ax1, ay1, ax2, ay2 = adj_box
                    
                    crop = frame[ay1:ay2, ax1:ax2]
                    
                    # [KOR] 2. 레터박스
                    # [ENG] 2. Letterbox
                    final_img = letterbox_resize(crop, (target_w, target_h))
                    
                    tw['writer'].write(final_img)
        
        # ==========================================
        # [KOR] 메타데이터용 BBox 수집
        # [ENG] Collect BBoxes for Metadata
        # ==========================================
        tube_bboxes = defaultdict(list)
        for f_idx, dets in enumerate(video_detections):
            for tid, box in dets.items():
                if tid in valid_tracks:
                    # [KOR] box는 (x1, y1, x2, y2)
                    # [ENG] box is (x1, y1, x2, y2)
                    tube_bboxes[tid].append({
                        "frame_idx": f_idx,
                        "box": list(box)
                    })

        # ==========================================
        # [KOR] 정리
        # [ENG] Clean up
        # ==========================================
        metadata = {}
        for tid, data in tube_writers.items():
            data['writer'].release()
            stats = valid_tracks[tid]
            metadata[f"id_{tid}"] = {
                "start_frame": stats['start_frame'],
                "end_frame": stats['end_frame'],
                "width": stats['max_w'],
                "height": stats['max_h'],
                "bboxes": tube_bboxes[tid]
            }
        
        cap.release()
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"        Tubes: {len(tube_writers)} extracted")

    def infer(self, video_path, save_dir):
        """
        [KOR] 메인 추론 파이프라인 실행
        [ENG] Run main inference pipeline
        
        Args:
            video_path: [KOR] 입력 비디오 경로 / [ENG] Path to input video
            save_dir: [KOR] 결과 저장 디렉토리 / [ENG] Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # [KOR] 이전 결과 정리
        # [ENG] Clean up previous results
        track_dir = os.path.join(save_dir, "track")
        tubes_dir = os.path.join(save_dir, "tubes")
        if os.path.exists(track_dir):
            shutil.rmtree(track_dir)
        if os.path.exists(tubes_dir):
            shutil.rmtree(tubes_dir)

        self._detect(video_path, save_dir)  # [KOR] 선택사항, 주로 시각적 확인용 / [ENG] Optional, mostly for visual check
        self._track(video_path, save_dir)
        self._extract_tubes(video_path, save_dir)