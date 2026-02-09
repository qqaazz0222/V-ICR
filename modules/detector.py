# [KOR] YOLO와 ByteTrack을 사용한 사람 탐지 및 추적
# [ENG] Person detection and tracking using YOLO and ByteTrack

import os
import cv2
import shutil
import numpy as np
import torch
import urllib.request
import sys

# [KOR] CoTracker 모듈 경로 추가
# [ENG] Add CoTracker module path
if os.path.join(os.getcwd(), 'modules') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'modules'))

try:
    from cotracker.predictor import CoTrackerOnlinePredictor
    from cotracker.utils.visualizer import Visualizer
except ImportError:
    # [KOR] 경로 문제로 실패 시, 직접 경로 설정 시도
    # [ENG] Try explicit path if import fails
    from modules.cotracker.predictor import CoTrackerOnlinePredictor
    from modules.cotracker.utils.visualizer import Visualizer

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

    def _download_cotracker_checkpoint(self, checkpoint_path):
        """
        [KOR] CoTracker 체크포인트 다운로드
        [ENG] Download CoTracker checkpoint
        """
        if not os.path.exists(checkpoint_path):
            print(f"    [!] Downloading CoTracker checkpoint to {checkpoint_path}...")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            url = "https://huggingface.co/facebook/cotracker/resolve/main/scaled_online.pth"
            try:
                urllib.request.urlretrieve(url, checkpoint_path)
                print("    [!] Download complete.")
            except Exception as e:
                print(f"    [!] Failed to download checkpoint: {str(e)}")

    def _run_cotracker(self, video_path, device='cuda'):
        """
        [KOR] CoTracker (Online) 실행하여 궤적 추출
        [ENG] Run CoTracker (Online) to extract trajectories
        
        Args:
            video_path: [KOR] 비디오 경로 / [ENG] Video path
            device: [KOR] 실행 장치 / [ENG] Execution device
            
        Returns:
            pred_tracks (np.array): (T, N, 2) 궤적 좌표
            pred_visibility (np.array): (T, N) 가시성 여부
        """
        checkpoint_path = "./checkpoints/cotracker_scaled_online.pth"
        self._download_cotracker_checkpoint(checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            print("    [!] CoTracker checkpoint missing, skipping.")
            return None, None
            
        try:
            print(" ├─[>] Running CoTracker (Online)...")
            
            # [KOR] Predictor 초기화
            # [ENG] Init Predictor
            # Checkpoint is loaded inside CoTrackerOnlinePredictor if passed
            model = CoTrackerOnlinePredictor(checkpoint=checkpoint_path)
            if torch.cuda.is_available() and device == 'cuda':
                model = model.cuda()
            
            cap = cv2.VideoCapture(video_path)
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # [KOR] 긴 축 기준 960으로 리사이즈 (cotracker 권장 또는 메모리 절약)
            # [ENG] Resize to max dim 960
            max_dim = 960
            scale = 1.0
            if max(W, H) > max_dim:
                scale = max_dim / max(W, H)
                new_W, new_H = int(W * scale), int(H * scale)
            else:
                new_W, new_H = W, H
                
            window_frames = []
            
            # [KOR] Step 처리 함수 정의 (online_demo.py 기반)
            # [ENG] Define step processing function (based on online_demo.py)
            def _process_step(window_frames, is_first_step):
                # We need RGB frames here
                video_chunk = (
                    torch.tensor(
                        np.stack(window_frames[-model.step * 2 :]), device=device
                    )
                    .float()
                    .permute(0, 3, 1, 2)[None]
                )  # (1, T, 3, H, W)
                
                return model(
                    video_chunk,
                    is_first_step=is_first_step,
                    grid_size=30, # [KOR] Grid Size = 30
                    grid_query_frame=0,
                )

            is_first_step = True
            idx = 0
            
            pred_tracks = None
            pred_visibility = None
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # [KOR] 리사이즈 및 BGR -> RGB 변환
                # [ENG] Resize and BGR -> RGB
                if scale != 1.0:
                    frame = cv2.resize(frame, (new_W, new_H))
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                window_frames.append(frame_rgb)
                
                if idx % model.step == 0 and idx != 0:
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step
                    )
                    is_first_step = False
                
                idx += 1
            
            cap.release()
            
            # [KOR] 남은 프레임 처리
            # [ENG] Process remaining frames
            if window_frames and (idx % model.step != 0 or is_first_step):
                 # [KOR] online_demo.py 로직 복제
                 # [ENG] Replicate online_demo.py logic
                 i = idx - 1
                 slice_start = -(i % model.step) - model.step - 1
                 
                 if abs(slice_start) > len(window_frames):
                     chunk = window_frames
                 else:
                     chunk = window_frames[slice_start:]
                 
                 video_chunk = (
                    torch.tensor(np.stack(chunk), device=device)
                    .float()
                    .permute(0, 3, 1, 2)[None]
                 )
                 
                 pred_tracks, pred_visibility = model(
                    video_chunk,
                    is_first_step=is_first_step,
                    grid_size=30,
                    grid_query_frame=0
                 )
                 
                 del video_chunk, chunk

            # [KOR] 메모리 정리
            # [ENG] Memory cleanup
            del window_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if pred_tracks is None:
                return None, None
                
            # [KOR] 결과 변환 및 스케일 복원
            # [ENG] Convert results and restore scale
            tracks = pred_tracks[0].cpu().numpy()
            vis = pred_visibility[0].cpu().numpy()
            
            # [KOR] GPU 텐서 정리
            # [ENG] Cleanup GPU tensors
            del pred_tracks, pred_visibility, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if scale != 1.0:
                tracks = tracks / scale
                
            return tracks, vis
            
        except Exception as e:
            print(f"    [!] CoTracker failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
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
            
        print(" ├─[>] Extracting tubes...")
        
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
        
        print(f" │  └─ Tracks: {len(valid_tracks)} valid / {len(track_stats)} total")
        
        # ==========================================
        # [KOR] CoTracker 실행 및 튜브별 궤적 필터링
        # [ENG] Run CoTracker and Filter Tracks per Tube
        # ==========================================
        cotracker_tracks, cotracker_vis = None, None
        tube_trajectories = {} # tid -> {tracks, vis}
        
        if valid_tracks:
            cotracker_tracks, cotracker_vis = self._run_cotracker(video_path)
            
            if cotracker_tracks is not None:
                print(" ├─[>] Filtering trajectories for tubes...")
                # [KOR] 각 튜브별로 관련 궤적 필터링
                # [ENG] Filter relevant trajectories for each tube
                T, N, _ = cotracker_tracks.shape
                
                for tid in valid_tracks:
                    meta = metadata_skeleton = {
                        "start": valid_tracks[tid]['start_frame'],
                        "end": valid_tracks[tid]['end_frame']
                    }
                    
                    # [KOR] 튜브의 BBox 정보 가져오기 (이미 계산됨)
                    # [ENG] Get tube BBox info (already computed)
                    # Use a simpler check: check intersection at middle frame or subsampled frames
                    
                    relevant_indices = []
                    
                    # [KOR] 튜브가 존재하는 프레임 범위
                    # [ENG] Tube existence frame range
                    start_f = int(meta["start"])
                    end_f = int(meta["end"])
                    
                    # [KOR] 튜브 BBox 맵핑 (프레임 -> 박스)
                    # [ENG] Tube BBox mapping
                    # Note: We haven't filled tube_bboxes yet fully?
                    # Actually we need to reconstruct tube bboxes from video_detections
                    # Or just check quickly
                    
                    # [KOR] 궤적 점수 계산: 튜브 박스 내부에 존재하는 프레임 수
                    # [ENG] Trajectory scoring: count frames inside tube box
                    track_scores = np.zeros(N, dtype=int)
                    
                    for f in range(start_f, end_f + 1, 5): # Check every 5th frame for speed
                        if f >= len(video_detections): continue
                        if tid not in video_detections[f]: continue
                        
                        box = video_detections[f][tid] #(x1, y1, x2, y2)
                        bx1, by1, bx2, by2 = box
                        
                        # [KOR] 현재 프레임의 모든 트랙 포인트
                        # [ENG] All track points at current frame
                        # tracks[f, :, 0] is x, tracks[f, :, 1] is y
                        pts_x = cotracker_tracks[f, :, 0]
                        pts_y = cotracker_tracks[f, :, 1]
                        vis = cotracker_vis[f, :]
                        
                        # [KOR] 박스 내부 체크
                        # [ENG] Check inside box
                        inside = (pts_x >= bx1) & (pts_x <= bx2) & (pts_y >= by1) & (pts_y <= by2) & vis
                        track_scores[inside] += 1
                        
                    # [KOR] 임계값: 튜브 지속시간의 20% 이상 내부에 존재 (또는 최소 3프레임)
                    # [ENG] Threshold: inside for >20% of tube duration (or min 3 checks)
                    # Since we checked every 5th frame, threshold is small
                    duration_checks = (end_f - start_f) // 5 + 1
                    threshold = max(1, duration_checks * 0.2)
                    
                    valid_indices = np.where(track_scores >= threshold)[0]
                    
                    if len(valid_indices) > 0:
                        tube_trajectories[tid] = {
                            "indices": valid_indices,
                            "tracks": cotracker_tracks[:, valid_indices, :], # (T, M, 2)
                            "vis": cotracker_vis[:, valid_indices]      # (T, M)
                        }
        
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
                    
                    final_img = letterbox_resize(crop, (target_w, target_h))
                    
                    # [KOR] 3. CoTracker 궤적 시각화
                    # [ENG] 3. Visualize CoTracker Trajectories
                    if cotracker_tracks is not None and tid in tube_trajectories:
                         traj_data = tube_trajectories[tid]
                         t_indices = traj_data["indices"]
                         
                         # [KOR] 현재 프레임에서 활성화된(보이는) 트랙만 시각화?
                         # [KOR] 아니면 궤적 전체? 보통 Trail을 그림.
                         # [ENG] Visualize only active (visible) tracks? Or trail.
                         
                         # Check visibility at current frame
                         vis_mask = cotracker_vis[frame_idx, t_indices] # (M,) bool
                         
                         if np.any(vis_mask):
                             # Draw trail for visible tracks
                             active_indices = np.where(vis_mask)[0]
                             
                             trail_len = 20
                             start_t = max(0, frame_idx - trail_len)
                             
                             # Get segments: (L, K, 2)
                             # t_indices[active_indices] maps to original global indices
                             # But we stored sub-set in 'tracks' key of tube_trajectories?
                             # No, tube_trajectories['tracks'] is (T, M, 2).
                             # So we take [start_t:frame_idx+1, active_indices, :]
                             
                             local_tracks = traj_data['tracks'] # (T, M, 2)
                             
                             # Helper to transform coordinates
                             # We need the params used in letterbox_resize
                             # calculate them again to be sure
                             cw, ch = crop.shape[1], crop.shape[0] # Crop size
                             # Note: crop came from adjust_bbox_to_aspect, so it matches aspect roughly
                             
                             scale = min(target_w / cw, target_h / ch)
                             nw, nh = int(cw * scale), int(ch * scale)
                             x_off = (target_w - nw) // 2
                             y_off = (target_h - nh) // 2
                             
                             pts_segment = local_tracks[start_t:frame_idx+1, active_indices] # (L, K, 2)
                             
                             # Transform
                             # Frame coords -> Crop coords -> Resize coords -> Final coords
                             # Box is (ax1, ay1, ax2, ay2)
                             
                             # X_final = (X_global - ax1) * scale + x_off
                             pts_final = np.zeros_like(pts_segment)
                             pts_final[..., 0] = (pts_segment[..., 0] - ax1) * scale + x_off
                             pts_final[..., 1] = (pts_segment[..., 1] - ay1) * scale + y_off
                             
                             pts_final = pts_final.astype(np.int32)
                             
                             # Draw polylines
                             # cv2.polylines expects list of arrays, each (L, 1, 2)
                             # We have (L, K, 2). Transpose to (K, L, 2) first
                             pts_final = pts_final.transpose(1, 0, 2) # (K, L, 2)
                             
                             polys = [p.reshape(-1, 1, 2) for p in pts_final]
                             
                             # Random colors or fixed?
                             # Let's use a rainbow dependent on track index or simple cyan
                             cv2.polylines(final_img, polys, isClosed=False, color=(0, 255, 255), thickness=2)
                             # Draw heads
                             for p in pts_final:
                                 if len(p) > 0:
                                     cv2.circle(final_img, tuple(p[-1]), 3, (0, 0, 255), -1)
                    
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
            
            # [KOR] Motion Features 저장
            # [ENG] Save Motion Features
            if tid in tube_trajectories:
                motion_path = os.path.join(output_dir, f"motion_features_{tid}.npy")
                # Save as a dictionary
                np.save(motion_path, {
                    "tracks": tube_trajectories[tid]["tracks"], # (T, M, 2)
                    "vis": tube_trajectories[tid]["vis"],       # (T, M)
                    "indices": tube_trajectories[tid]["indices"]
                })
        
        cap.release()
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f" │  └─ Tubes: {len(tube_writers)} extracted")

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