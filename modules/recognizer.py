# [KOR] Qwen3-VL을 활용한 시간적 soft-label 정제 기반 행동 인식
# [ENG] Action recognition using Qwen3-VL with temporal soft-label refinement

import os
import re
import json
import cv2
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class Recognizer:
    """
    [KOR] Qwen3-VL을 활용한 시간적 soft-label 정제 기반 행동 인식기
    [ENG] Action recognizer using Qwen3-VL with temporal soft-label refinement
    
    [KOR] 이 클래스는 VLM을 사용하여 사람 튜브에서 1초 단위 행동 라벨링을 수행하고,
          5개의 후보를 가진 soft-label을 생성하여 반복적으로 정제합니다.
    [ENG] This class performs per-second action labeling on person tubes using a VLM,
          generating soft-labels with top-5 candidates and iteratively refining them.
    
    Features:
    - [KOR] Phase 0: 대표 튜브 샘플링 및 자동 labelmap 생성 / [ENG] Phase 0: Representative tube sampling and automatic labelmap generation
    - [KOR] 1초 간격 분석 / [ENG] 1-second interval analysis
    - [KOR] 초당 5개 후보 soft-label / [ENG] Soft-labels with 5 candidates per second
    - [KOR] Feature 기반 군집화 정제 / [ENG] Feature-based clustering refinement
    - [KOR] labelmap 행동만 사용한 제한된 정제 / [ENG] Constrained refinement using only labelmap actions
    
    [KOR] 출력 형식: id-time-action 트리플
    [ENG] Output format: id-time-action triplets
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        """
        [KOR] Qwen3-VL 모델로 인식기 초기화
        [ENG] Initialize the recognizer with Qwen3-VL model
        
        Args:
            model_name: [KOR] HuggingFace 모델 식별자 / [ENG] HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """
        [KOR] Qwen3-VL 모델과 프로세서 로드
        [ENG] Load the Qwen3-VL model and processor
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # [KOR] 비디오 정보 캐시
        # [ENG] Video info cache
        self._video_info_cache: Dict[str, Tuple[float, int, int]] = {}
    
    def _get_video_info(self, video_path: str) -> Tuple[float, int, int]:
        """
        [KOR] 비디오 FPS, 총 프레임 수, 초 단위 길이 반환 (캐시 적용)
        [ENG] Get video FPS, total frames, and duration in seconds (with caching)
        
        Returns:
            [KOR] (fps, 총_프레임수, 초_단위_길이) 튜플
            [ENG] Tuple of (fps, total_frames, duration_seconds)
        """
        # [KOR] 캐시 확인
        # [ENG] Check cache
        if video_path in self._video_info_cache:
            return self._video_info_cache[video_path]
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps <= 0:
            fps = 30.0
        duration = int(total_frames / fps)
        
        # [KOR] 캐시 저장
        # [ENG] Store cache
        result = (fps, total_frames, duration)
        self._video_info_cache[video_path] = result
        return result
    
    def _extract_frames_for_second(self, video_path: str, second: int, 
                                    fps: float, num_frames: int = 4) -> List[np.ndarray]:
        """
        [KOR] 비디오의 특정 초에서 프레임 추출
        [ENG] Extract frames for a specific second of video
        
        Args:
            video_path: [KOR] 비디오 파일 경로 / [ENG] Path to video file
            second: [KOR] 대상 초 (0-indexed) / [ENG] Target second (0-indexed)
            fps: [KOR] 비디오 FPS / [ENG] Video FPS
            num_frames: [KOR] 이 초에서 추출할 프레임 수 / [ENG] Number of frames to extract from this second
            
        Returns:
            [KOR] numpy 배열 형태의 프레임 리스트 (RGB 형식)
            [ENG] List of frames as numpy arrays (RGB format)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        start_frame = int(second * fps)
        end_frame = int((second + 1) * fps)
        frame_interval = max(1, (end_frame - start_frame) // num_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        count = 0
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = start_frame + count
            if current_frame >= end_frame:
                break
            if count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        
        cap.release()
        return frames
    
    def _extract_frames(self, video_path: str, fps: float = 1.0, max_frames: int = 16) -> List[np.ndarray]:
        """
        [KOR] 지정된 FPS로 비디오에서 프레임 추출
        [ENG] Extract frames from video at specified FPS
        
        Args:
            video_path: [KOR] 비디오 파일 경로 / [ENG] Path to video file
            fps: [KOR] 추출할 초당 프레임 수 / [ENG] Target frames per second to extract
            max_frames: [KOR] 추출할 최대 프레임 수 / [ENG] Maximum number of frames to extract
            
        Returns:
            [KOR] numpy 배열 형태의 프레임 리스트 (RGB 형식)
            [ENG] List of frames as numpy arrays (RGB format)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps <= 0:
            cap.release()
            return frames
        
        frame_interval = max(1, int(video_fps / fps))
        
        expected_frames = total_frames // frame_interval
        if expected_frames > max_frames:
            frame_interval = total_frames // max_frames
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(frames) >= max_frames:
                    break
            count += 1
        
        cap.release()
        return frames
    
    def _inference(self, frames: List[np.ndarray], prompt: str) -> str:
        """
        [KOR] 주어진 프롬프트로 프레임에 대해 추론 실행
        [ENG] Run inference on frames with given prompt
        
        Args:
            frames: [KOR] 프레임 리스트 (RGB 형식) / [ENG] List of frames (RGB format)
            prompt: [KOR] 모델용 텍스트 프롬프트 / [ENG] Text prompt for the model
            
        Returns:
            [KOR] 모델 출력 텍스트 / [ENG] Model output text
        """
        content = []
        for frame in frames:
            content.append({
                "type": "image",
                "image": frame,
            })
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [{"role": "user", "content": content}]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # [KOR] GPU 메모리 최적화: 추론 전 캐시 정리
        # [ENG] GPU memory optimization: clear cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # [KOR] 추론 후 메모리 정리
        # [ENG] Clear memory after inference
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output[0]
    
    def _parse_json_response(self, response: str) -> Any:
        """
        [KOR] 모델 응답에서 JSON 파싱
        [ENG] Parse JSON from model response
        
        Args:
            response: [KOR] 원본 모델 출력 / [ENG] Raw model output
            
        Returns:
            [KOR] 파싱된 JSON (dict 또는 list) / [ENG] Parsed JSON (dict or list)
        """
        # [KOR] JSON 블록 찾기 시도
        # [ENG] Try to find JSON block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # [KOR] JSON 배열 또는 객체 추출 시도
        # [ENG] Try to extract JSON array or object
        json_match = re.search(r'(\[[\s\S]*?\]|\{[\s\S]*?\})', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # [KOR] 전체 응답 시도
        # [ENG] Try full response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_response": response}
    
    def _analyze_second_initial(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        [KOR] 1초 세그먼트의 초기 분석, soft-label 반환
        [ENG] Initial analysis of a 1-second segment, returning soft-labels
        
        Args:
            frames: [KOR] 이 초의 프레임들 / [ENG] Frames from this second
            
        Returns:
            [KOR] 신뢰도 점수가 포함된 5개 행동 후보 리스트
            [ENG] List of 5 action candidates with confidence scores
        """
        if not frames:
            return [{"action": "unknown", "confidence": 0.2}] * 5
        
        prompt = """Analyze these consecutive frames showing a person's action during 1 second.
Identify the most likely actions being performed.

Output a JSON array of exactly 5 action candidates, ranked by likelihood:
[
  {"action": "action_name", "confidence": 0.95},
  {"action": "action_name", "confidence": 0.80},
  {"action": "action_name", "confidence": 0.60},
  {"action": "action_name", "confidence": 0.40},
  {"action": "action_name", "confidence": 0.20}
]

Rules:
- Use common action names (walking, standing, sitting, running, talking, gesturing, etc.)
- Confidence scores must sum to approximately 1.0
- Be specific but concise in action names
- Return ONLY the JSON array."""

        response = self._inference(frames, prompt)
        result = self._parse_json_response(response)
        
        # [KOR] 검증 및 정규화
        # [ENG] Validate and normalize
        if isinstance(result, list) and len(result) >= 5:
            candidates = []
            for item in result[:5]:
                if isinstance(item, dict) and "action" in item:
                    candidates.append({
                        "action": str(item.get("action", "unknown")).lower().strip(),
                        "confidence": float(item.get("confidence", 0.2))
                    })
            if len(candidates) == 5:
                return candidates
        
        # [KOR] 폴백
        # [ENG] Fallback
        return [{"action": "unknown", "confidence": 0.2}] * 5
    
    def _analyze_second_with_predefined_labels(self, frames: List[np.ndarray], 
                                                labelmap: List[str]) -> List[Dict[str, Any]]:
        """
        [KOR] 사전 정의된 labelmap을 사용하여 1초 세그먼트 분석
        [ENG] Analyze 1-second segment using predefined labelmap
        
        Args:
            frames: [KOR] 이 초의 프레임들 / [ENG] Frames from this second
            labelmap: [KOR] 사전 정의된 행동 라벨 리스트 / [ENG] Predefined action labels list
            
        Returns:
            [KOR] 신뢰도 점수가 포함된 5개 행동 후보 리스트 (labelmap 내에서만)
            [ENG] List of 5 action candidates with confidence scores (only from labelmap)
        """
        if not frames or not labelmap:
            return [{"action": labelmap[0] if labelmap else "unknown", "confidence": 0.2}] * 5
        
        labels_str = ", ".join(labelmap)
        
        prompt = f"""Analyze these consecutive frames showing a person's action during 1 second.
Identify the most likely actions being performed from ONLY the following allowed labels.

Allowed action labels: [{labels_str}]

Output a JSON array of exactly 5 action candidates from the allowed labels, ranked by likelihood:
[
  {{"action": "label_from_list", "confidence": 0.95}},
  {{"action": "label_from_list", "confidence": 0.80}},
  {{"action": "label_from_list", "confidence": 0.60}},
  {{"action": "label_from_list", "confidence": 0.40}},
  {{"action": "label_from_list", "confidence": 0.20}}
]

Rules:
- ONLY use actions from the allowed labels list above
- Confidence scores must sum to approximately 1.0
- Return ONLY the JSON array."""

        response = self._inference(frames, prompt)
        result = self._parse_json_response(response)
        
        # [KOR] 검증 및 정규화
        # [ENG] Validate and normalize
        labelmap_lower = [l.lower() for l in labelmap]
        if isinstance(result, list) and len(result) >= 1:
            candidates = []
            for item in result[:5]:
                if isinstance(item, dict) and "action" in item:
                    action = str(item.get("action", "")).lower().strip()
                    # [KOR] 행동이 labelmap에 있는지 검증
                    # [ENG] Validate action is in labelmap
                    if action in labelmap_lower:
                        candidates.append({
                            "action": action,
                            "confidence": float(item.get("confidence", 0.2))
                        })
            
            # [KOR] 필요한 경우 나머지 labelmap 행동으로 채우기
            # [ENG] Pad with remaining labelmap actions if needed
            if len(candidates) < 5:
                used = {c["action"] for c in candidates}
                for label in labelmap:
                    if label.lower() not in used and len(candidates) < 5:
                        candidates.append({"action": label.lower(), "confidence": 0.1})
            
            if candidates:
                return candidates[:5]
        
        # [KOR] 폴백: labelmap 전체에 균등 분배
        # [ENG] Fallback: distribute evenly across labelmap
        return [{"action": l.lower(), "confidence": 1.0/len(labelmap)} for l in labelmap[:5]]

    
    def _group_similar_actions(self, all_actions: List[str], video_path: str) -> Dict[str, List[str]]:
        """
        [KOR] VLM을 사용하여 유사한 행동을 대표 카테고리로 그룹화
        [ENG] Group similar actions into representative categories using VLM
        
        Args:
            all_actions: [KOR] 감지된 모든 행동 이름 리스트 / [ENG] List of all detected action names
            video_path: [KOR] 컨텍스트용 비디오 경로 / [ENG] Path to video for context
            
        Returns:
            [KOR] 대표 행동을 유사 행동 리스트에 매핑하는 딕셔너리
            [ENG] Dictionary mapping representative action to list of similar actions
        """
        if not all_actions:
            return {}
        
        # [KOR] 중복 제거 및 정렬
        # [ENG] Remove duplicates and sort
        unique_actions = sorted(list(set(all_actions)))
        
        if len(unique_actions) <= 5:
            # [KOR] 이미 충분히 작음, 그룹화 불필요
            # [ENG] Already small enough, no grouping needed
            return {action: [action] for action in unique_actions}
        
        # [KOR] 컨텍스트 프레임 가져오기
        # [ENG] Get some context frames
        frames = self._extract_frames(video_path, fps=0.5, max_frames=8)
        
        if not frames:
            return {action: [action] for action in unique_actions}
        
        actions_str = ", ".join(unique_actions)
        
        prompt = f"""Given this video context and the following detected actions, group similar actions into representative categories.

Detected actions: [{actions_str}]

Rules for grouping:
1. Group actions that represent the same or very similar movements
2. For repetitive patterns (e.g., "sitting down" + "standing up" repeatedly), use a compound action name like "squatting" or "sit-stand exercise"
3. Keep distinct actions separate
4. Use clear, concise representative names
5. Aim for 5-10 final categories maximum

Output a JSON object where keys are representative action names and values are arrays of the original actions they represent:
{{
  "representative_action": ["original_action1", "original_action2"],
  "another_action": ["original_action3"]
}}

Return ONLY the JSON object."""

        response = self._inference(frames, prompt)
        result = self._parse_json_response(response)
        
        if isinstance(result, dict) and not result.get("raw_response"):
            # [KOR] 모든 원본 행동이 포함되었는지 검증
            # [ENG] Validate that all original actions are covered
            covered = set()
            valid_groups = {}
            for rep_action, original_actions in result.items():
                if isinstance(original_actions, list):
                    valid_originals = [a for a in original_actions if a in unique_actions]
                    if valid_originals:
                        valid_groups[rep_action.lower().strip()] = valid_originals
                        covered.update(valid_originals)
            
            # [KOR] 그룹화되지 않은 행동을 자체 카테고리로 추가
            # [ENG] Add ungrouped actions as their own category
            for action in unique_actions:
                if action not in covered:
                    valid_groups[action] = [action]
            
            return valid_groups
        
        # [KOR] 폴백: 각 행동을 자체 카테고리로
        # [ENG] Fallback: each action is its own category
        return {action: [action] for action in unique_actions}
    
    def _create_labelmap(self, action_groups: Dict[str, List[str]], save_dir: str) -> List[str]:
        """
        [KOR] 행동 그룹에서 labelmap 생성 및 저장
        [ENG] Create and save labelmap from action groups
        
        Args:
            action_groups: [KOR] 대표 행동을 원본 행동에 매핑하는 딕셔너리 / [ENG] Dictionary of representative actions to original actions
            save_dir: [KOR] label_map.txt 저장 디렉토리 / [ENG] Directory to save label_map.txt
            
        Returns:
            [KOR] 대표 행동 라벨 리스트 / [ENG] List of representative action labels
        """
        labels = sorted(action_groups.keys())
        
        # [KOR] labelmap 저장
        # [ENG] Save labelmap
        labelmap_path = os.path.join(save_dir, "label_map.txt")
        with open(labelmap_path, "w", encoding="utf-8") as f:
            for i, label in enumerate(labels):
                originals = action_groups[label]
                f.write(f"{i}: {label}\n")
                f.write(f"   -> {', '.join(originals)}\n")
        
        return labels
    
    def _map_action_to_label(self, action: str, action_groups: Dict[str, List[str]]) -> str:
        """
        [KOR] 원본 행동을 대표 라벨에 매핑
        [ENG] Map an original action to its representative label
        
        Args:
            action: [KOR] 원본 행동 이름 / [ENG] Original action name
            action_groups: [KOR] 대표 행동을 원본 행동에 매핑하는 딕셔너리 / [ENG] Dictionary of representative actions to original actions
            
        Returns:
            [KOR] 대표 라벨 이름 / [ENG] Representative label name
        """
        action_lower = action.lower().strip()
        for rep_label, originals in action_groups.items():
            if action_lower in [o.lower().strip() for o in originals]:
                return rep_label
        return action_lower
    
    def _refine_second_with_labelmap(self, frames: List[np.ndarray],
                                      current_candidates: List[Dict],
                                      labelmap: List[str],
                                      prev_action: Optional[str] = None,
                                      next_action: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        [KOR] labelmap 제약 조건을 사용하여 soft-label 정제
        [ENG] Refine soft-labels using labelmap constraints
        
        Args:
            frames: [KOR] 이 초의 프레임들 / [ENG] Frames from this second
            current_candidates: [KOR] 현재 soft-label 후보 / [ENG] Current soft-label candidates
            labelmap: [KOR] 허용된 행동 라벨 리스트 / [ENG] List of allowed action labels
            prev_action: [KOR] 이전 초의 최고 행동 (선택사항) / [ENG] Previous second's top action (optional)
            next_action: [KOR] 다음 초의 최고 행동 (선택사항) / [ENG] Next second's top action (optional)
            
        Returns:
            [KOR] labelmap에서만 선택된 5개 행동 후보의 정제된 리스트
            [ENG] Refined list of 5 action candidates from labelmap only
        """
        if not frames or not labelmap:
            return current_candidates
        
        context_parts = []
        
        if prev_action:
            context_parts.append(f"Previous second: {prev_action}")
        if next_action:
            context_parts.append(f"Next second: {next_action}")
        
        current_top = current_candidates[0]["action"] if current_candidates else "unknown"
        context_parts.append(f"Initial prediction: {current_top}")
        
        context = "\n".join(context_parts) if context_parts else "No temporal context"
        labels_str = ", ".join(labelmap)
        
        prompt = f"""Analyze these frames showing a person's action during 1 second.
Assign probability scores to ONLY the following action labels.

Allowed action labels: [{labels_str}]

Temporal context:
{context}

Output a JSON array of exactly 5 candidates from the allowed labels, with probabilities:
[
  {{"action": "label_from_list", "confidence": 0.95}},
  {{"action": "label_from_list", "confidence": 0.80}},
  ...
]

Rules:
- ONLY use actions from the allowed labels list
- Probabilities should reflect the likelihood of each action
- Consider temporal consistency with previous/next seconds
- Return ONLY the JSON array."""

        response = self._inference(frames, prompt)
        result = self._parse_json_response(response)
        
        if isinstance(result, list) and len(result) >= 1:
            candidates = []
            for item in result[:5]:
                if isinstance(item, dict) and "action" in item:
                    action = str(item.get("action", "")).lower().strip()
                    # [KOR] 행동이 labelmap에 있는지 검증
                    # [ENG] Validate action is in labelmap
                    if action in [l.lower() for l in labelmap]:
                        candidates.append({
                            "action": action,
                            "confidence": float(item.get("confidence", 0.2))
                        })
            
            # [KOR] 필요한 경우 나머지 labelmap 행동으로 채우기
            # [ENG] Pad with remaining labelmap actions if needed
            if len(candidates) < 5:
                used = {c["action"] for c in candidates}
                for label in labelmap:
                    if label.lower() not in used and len(candidates) < 5:
                        candidates.append({"action": label.lower(), "confidence": 0.1})
            
            if candidates:
                return candidates[:5]
        
        # [KOR] 폴백: labelmap 전체에 균등 분배
        # [ENG] Fallback: distribute evenly across labelmap
        return [{"action": l.lower(), "confidence": 1.0/len(labelmap)} for l in labelmap[:5]]
    
    # ==========================================
    # [KOR] Phase 0: Action Discovery (대표 행동 발견)
    # [ENG] Phase 0: Action Discovery
    # ==========================================
    
    def _extract_motion_features(self, tube_id: str, tubes_dir: str, metadata: Dict) -> np.ndarray:
        """
        [KOR] 튜브의 모션 특징 추출 (속도, 이동량, 크기 변화)
        [ENG] Extract motion features from tube (velocity, displacement, size change)
        
        Args:
            tube_id: [KOR] 튜브 ID / [ENG] Tube ID
            tubes_dir: [KOR] 튜브 디렉토리 / [ENG] Tubes directory
            metadata: [KOR] 메타데이터 / [ENG] Metadata
            
        Returns:
            [KOR] 모션 특징 벡터 (8차원) / [ENG] Motion feature vector (8-dim)
        """
        tube_meta = metadata.get(tube_id, {})
        bboxes = tube_meta.get("bboxes", [])
        
        if len(bboxes) < 2:
            return np.zeros(8)
        
        # [KOR] BBox 중심점 계산
        # [ENG] Calculate bbox centers
        centers = []
        sizes = []
        for bbox_entry in bboxes:
            box = bbox_entry["box"]  # [x1, y1, x2, y2]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]
            centers.append([cx, cy])
            sizes.append([w, h])
        
        centers = np.array(centers)
        sizes = np.array(sizes)
        
        # [KOR] 속도 계산 (프레임 간 이동)
        # [ENG] Calculate velocity (movement between frames)
        velocities = np.diff(centers, axis=0)
        
        # [KOR] 특징 추출
        # [ENG] Extract features
        features = [
            np.mean(np.abs(velocities[:, 0])) if len(velocities) > 0 else 0,  # mean x velocity
            np.mean(np.abs(velocities[:, 1])) if len(velocities) > 0 else 0,  # mean y velocity
            np.std(velocities[:, 0]) if len(velocities) > 0 else 0,           # x velocity std
            np.std(velocities[:, 1]) if len(velocities) > 0 else 0,           # y velocity std
            np.sum(np.sqrt(np.sum(velocities**2, axis=1))) if len(velocities) > 0 else 0,  # total displacement
            np.mean(sizes[:, 0]),                                              # mean width
            np.mean(sizes[:, 1]),                                              # mean height
            np.std(sizes[:, 0] * sizes[:, 1]) if len(sizes) > 1 else 0        # size variation
        ]
        
        return np.array(features)
    
    def _sample_representative_tubes(self, metadata: Dict, tubes_dir: str, 
                                      n_samples: int = 15) -> List[str]:
        """
        [KOR] K-Means를 사용하여 대표 튜브 샘플링
        [ENG] Sample representative tubes using K-Means clustering
        
        Args:
            metadata: [KOR] 메타데이터 / [ENG] Metadata
            tubes_dir: [KOR] 튜브 디렉토리 / [ENG] Tubes directory
            n_samples: [KOR] 샘플 수 / [ENG] Number of samples
            
        Returns:
            [KOR] 대표 튜브 ID 리스트 / [ENG] List of representative tube IDs
        """
        tube_ids = list(metadata.keys())
        
        if len(tube_ids) <= n_samples:
            return tube_ids
        
        if not SKLEARN_AVAILABLE:
            # [KOR] sklearn 없으면 균등 샘플링
            # [ENG] Uniform sampling if sklearn not available
            step = len(tube_ids) // n_samples
            return [tube_ids[i * step] for i in range(n_samples)]
        
        # [KOR] 모션 특징 추출
        # [ENG] Extract motion features
        features = []
        valid_ids = []
        for tube_id in tube_ids:
            feat = self._extract_motion_features(tube_id, tubes_dir, metadata)
            if np.sum(np.abs(feat)) > 0:  # 유효한 특징만
                features.append(feat)
                valid_ids.append(tube_id)
        
        if len(features) < n_samples:
            return valid_ids
        
        features = np.array(features)
        
        # [KOR] 정규화
        # [ENG] Normalize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # [KOR] K-Means 군집화
        # [ENG] K-Means clustering
        n_clusters = min(n_samples, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # [KOR] 각 클러스터에서 중심에 가장 가까운 튜브 선택
        # [ENG] Select tube closest to center in each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                cluster_features = features_scaled[cluster_mask]
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected.append(valid_ids[closest_idx])
        
        return selected
    
    def _caption_tube_freeform(self, tube_path: str) -> str:
        """
        [KOR] 튜브에 대해 자유 형식 캡셔닝 수행
        [ENG] Perform free-form captioning on a tube
        
        Args:
            tube_path: [KOR] 튜브 비디오 경로 / [ENG] Path to tube video
            
        Returns:
            [KOR] 행동 설명 캡션 / [ENG] Action description caption
        """
        fps, total_frames, duration = self._get_video_info(tube_path)
        
        if duration <= 0:
            return "unknown action"
        
        # [KOR] 전체 비디오에서 균등하게 프레임 추출
        # [ENG] Extract frames uniformly from entire video
        frames = self._extract_frames(tube_path, fps=2.0, max_frames=8)
        
        if not frames:
            return "unknown action"
        
        prompt = """Describe the action being performed by the person in these video frames.
Focus on:
1. The main physical action (e.g., walking, running, sitting, jumping)
2. Any objects being interacted with
3. The manner or style of the action

Provide a concise description in 5-10 words.
Example outputs: "walking slowly while carrying a bag", "doing push-ups on the floor", "waving hand in greeting"

Your response (action description only):"""

        response = self._inference(frames, prompt)
        
        # [KOR] 응답 정리
        # [ENG] Clean up response
        action = response.strip().lower()
        # 첫 문장만 사용
        action = action.split('.')[0].split('\n')[0]
        
        return action if action else "unknown action"
    
    def _discover_action_vocabulary(self, metadata: Dict, tubes_dir: str, 
                                     video_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        [KOR] Phase 0: 대표 튜브 분석으로 행동 어휘 발견
        [ENG] Phase 0: Discover action vocabulary by analyzing representative tubes
        
        Args:
            metadata: [KOR] 메타데이터 / [ENG] Metadata
            tubes_dir: [KOR] 튜브 디렉토리 / [ENG] Tubes directory
            video_path: [KOR] 원본 비디오 경로 / [ENG] Original video path
            
        Returns:
            [KOR] (labelmap, action_groups) 튜플 / [ENG] Tuple of (labelmap, action_groups)
        """
        print(" │  ├─ Phase 0: Discovering action vocabulary...")
        
        # [KOR] 1. 대표 튜브 샘플링
        # [ENG] 1. Sample representative tubes
        representative_tubes = self._sample_representative_tubes(metadata, tubes_dir, n_samples=15)
        print(f" │  │  └─ Sampled {len(representative_tubes)} representative tubes")
        
        # [KOR] 2. 자유 형식 캡셔닝
        # [ENG] 2. Free-form captioning
        captions = []
        for tube_id in representative_tubes:
            tube_path = os.path.join(tubes_dir, f"{tube_id}.mp4")
            if os.path.exists(tube_path):
                caption = self._caption_tube_freeform(tube_path)
                if caption and caption != "unknown action":
                    captions.append(caption)
        
        if not captions:
            # [KOR] 폴백: 기본 행동 세트
            # [ENG] Fallback: default action set
            return ["standing", "walking", "sitting", "unknown"], {"standing": ["standing"], "walking": ["walking"], "sitting": ["sitting"], "unknown": ["unknown"]}
        
        # [KOR] 3. LLM으로 캡션 요약 및 카테고리화
        # [ENG] 3. Summarize and categorize captions with LLM
        captions_str = "\n".join([f"- {c}" for c in captions])
        
        # [KOR] 컨텍스트 프레임 가져오기
        # [ENG] Get context frames
        frames = self._extract_frames(video_path, fps=0.5, max_frames=4)
        
        prompt = f"""Based on these action descriptions observed in a video:
{captions_str}

Create a concise action vocabulary (label map) that:
1. Groups similar actions into categories
2. Uses simple, clear action names
3. Has 5-15 categories maximum
4. Each category name should be 1-3 words

Output a JSON object where keys are category names and values are lists of original descriptions that belong to that category:
{{
  "walking": ["walking slowly", "moving forward"],
  "sitting": ["sitting down", "seated"],
  ...
}}

Return ONLY the JSON object."""

        response = self._inference(frames, prompt)
        result = self._parse_json_response(response)
        
        if isinstance(result, dict) and not result.get("raw_response"):
            # [KOR] 유효한 그룹
            # [ENG] Valid groups
            action_groups = {}
            for category, descriptions in result.items():
                cat_name = str(category).lower().strip()
                if isinstance(descriptions, list):
                    action_groups[cat_name] = [str(d).lower().strip() for d in descriptions]
                else:
                    action_groups[cat_name] = [cat_name]
            
            labelmap = sorted(action_groups.keys())
            return labelmap, action_groups
        
        # [KOR] 폴백: 직접 그룹화
        # [ENG] Fallback: group directly using existing method
        action_groups = self._group_similar_actions(captions, video_path)
        labelmap = sorted(action_groups.keys())
        return labelmap, action_groups
    
    def _analyze_tube_temporal(self, tube_path: str, video_path: str, 
                                working_dir: str, iterations: int = 2,
                                predefined_labelmap: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        [KOR] 초별 soft-label과 반복 정제로 튜브 비디오 분석
        [ENG] Analyze a tube video with per-second soft-labels and iterative refinement
        
        Args:
            tube_path: [KOR] 튜브 비디오 경로 / [ENG] Path to tube video
            video_path: [KOR] 컨텍스트용 원본 비디오 경로 / [ENG] Path to original video for context
            working_dir: [KOR] labelmap 저장용 작업 디렉토리 / [ENG] Working directory for saving labelmap
            iterations: [KOR] 정제 반복 횟수 / [ENG] Number of refinement iterations
            predefined_labelmap: [KOR] 사전 정의된 라벨 리스트 (선택사항) / [ENG] Predefined labels list (optional)
            
        Returns:
            [KOR] 초별 soft-label과 최종 행동이 포함된 딕셔너리
            [ENG] Dictionary with per-second soft-labels and final actions
        """
        fps, total_frames, duration = self._get_video_info(tube_path)
        
        if duration <= 0:
            return {"error": "Invalid video", "temporal_labels": [], "final_actions": []}
        
        # [KOR] Phase 1: 초기 초별 분석
        # [ENG] Phase 1: Initial per-second analysis
        temporal_labels = {}  # [KOR] 초 -> 5개 후보 리스트 / [ENG] second -> list of 5 candidates
        all_detected_actions = []
        
        for sec in range(duration):
            frames = self._extract_frames_for_second(tube_path, sec, fps, num_frames=4)
            
            # [KOR] 사전 정의된 labelmap이 있으면 해당 라벨만 사용
            # [ENG] Use predefined labelmap if available
            if predefined_labelmap:
                candidates = self._analyze_second_with_predefined_labels(frames, predefined_labelmap)
            else:
                candidates = self._analyze_second_initial(frames)
            
            temporal_labels[sec] = candidates
            
            # [KOR] 그룹화를 위해 감지된 모든 행동 수집
            # [ENG] Collect all detected actions for grouping
            for c in candidates:
                if c["action"] != "unknown":
                    all_detected_actions.append(c["action"])
        
        return {
            "duration": duration,
            "temporal_labels": temporal_labels,
            "detected_actions": all_detected_actions
        }
    
    def recognize(self, 
                  video_path: str, 
                  tubes_dir: str,
                  iterations: int = 2,
                  predefined_labelmap: Optional[List[str]] = None) -> Dict:
        """
        [KOR] 시간적 soft-label 정제를 포함한 메인 인식 파이프라인
        [ENG] Main recognition pipeline with temporal soft-label refinement
        
        Args:
            video_path: [KOR] 원본 전체 비디오 경로 / [ENG] Path to original full video
            tubes_dir: [KOR] 튜브 비디오와 메타데이터가 포함된 디렉토리 / [ENG] Directory containing tube videos and metadata
            iterations: [KOR] 정제 반복 횟수 (기본값: 2) / [ENG] Number of refinement iterations (default: 2)
            predefined_labelmap: [KOR] 사전 정의된 라벨 리스트 (선택사항) / [ENG] Predefined labels list (optional)
            
        Returns:
            [KOR] id-time-action 형식의 인식 결과 딕셔너리
            [ENG] Dictionary with recognition results in id-time-action format
        """
        print(" ├─[>] Recognizing actions...")
        
        # [KOR] 사전 정의된 labelmap 사용 여부 확인
        # [ENG] Check if using predefined labelmap
        use_predefined = predefined_labelmap is not None and len(predefined_labelmap) > 0
        if use_predefined:
            print(f" │  ├─ Using predefined label-map: {len(predefined_labelmap)} labels")
        
        # [KOR] 메타데이터 로드
        # [ENG] Load metadata
        metadata_path = os.path.join(os.path.dirname(tubes_dir), "metadata.json")
        if not os.path.exists(metadata_path):
            return {"error": "Metadata not found", "tubes": {}}
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # [KOR] 작업 디렉토리 가져오기 (tubes_dir의 부모)
        # [ENG] Get working directory (parent of tubes_dir)
        working_dir = os.path.dirname(tubes_dir)
        
        # ==========================================
        # [KOR] Phase 0: 행동 어휘 발견 (사전 정의 없을 때)
        # [ENG] Phase 0: Discover action vocabulary (when no predefined)
        # ==========================================
        if use_predefined:
            labelmap = [l.lower() for l in predefined_labelmap]
            action_groups = {l: [l] for l in labelmap}
            
            # [KOR] 사전 정의된 labelmap을 working 디렉토리에 저장
            # [ENG] Save predefined labelmap to working directory
            labelmap_path = os.path.join(working_dir, "label_map.txt")
            with open(labelmap_path, "w", encoding="utf-8") as f:
                for i, label in enumerate(labelmap):
                    f.write(f"{i}: {label}\n")
        else:
            # [KOR] Phase 0: 대표 튜브 기반 어휘 발견
            # [ENG] Phase 0: Discovery from representative tubes
            labelmap, action_groups = self._discover_action_vocabulary(
                metadata, tubes_dir, video_path
            )
            
            # [KOR] 발견된 labelmap 저장
            # [ENG] Save discovered labelmap
            labelmap_path = os.path.join(working_dir, "label_map.txt")
            with open(labelmap_path, "w", encoding="utf-8") as f:
                for i, label in enumerate(labelmap):
                    f.write(f"{i}: {label}\n")
        
        print(f" │  ├─ Label-Map: {len(labelmap)} action types")
        
        # ==========================================
        # [KOR] Phase 1: 확정된 labelmap으로 모든 튜브 분류
        # [ENG] Phase 1: Classify all tubes with confirmed labelmap
        # ==========================================
        print(f" │  ├─ Phase 1: Classifying {len(metadata)} tubes with labelmap...")
        tube_initial_results = {}
        all_detected_actions = []
        
        for tube_id in metadata.keys():
            tube_path = os.path.join(tubes_dir, f"{tube_id}.mp4")
            if os.path.exists(tube_path):
                # [KOR] labelmap을 predefined로 전달하여 객관식 분류
                # [ENG] Pass labelmap as predefined for multiple-choice classification
                result = self._analyze_tube_temporal(
                    tube_path, video_path, working_dir, iterations,
                    predefined_labelmap=labelmap
                )
                tube_initial_results[tube_id] = result
                all_detected_actions.extend(result.get("detected_actions", []))
        
        # ==========================================
        # [KOR] Phase 2: labelmap을 사용한 반복 정제
        # [ENG] Phase 2: Iterative refinement with labelmap
        # ==========================================
        if iterations > 1:
            for iteration in range(iterations - 1):
                print(f" │  ├─ Phase 2: Refinement iteration ({iteration + 1}/{iterations})...")
                
                for tube_id, result in tube_initial_results.items():
                    tube_path = os.path.join(tubes_dir, f"{tube_id}.mp4")
                    if not os.path.exists(tube_path):
                        continue
                    
                    fps, _, duration = self._get_video_info(tube_path)
                    temporal_labels = result.get("temporal_labels", {})
                    refined_labels = {}
                    
                    for sec in range(duration):
                        if sec not in temporal_labels:
                            continue
                        
                        frames = self._extract_frames_for_second(tube_path, sec, fps, num_frames=4)
                        
                        prev_action = None
                        next_action = None
                        
                        # [KOR] 시간적 컨텍스트 가져오기
                        # [ENG] Get temporal context
                        if sec - 1 in temporal_labels and temporal_labels[sec - 1]:
                            prev_action = temporal_labels[sec - 1][0]["action"]
                        if sec + 1 in temporal_labels and temporal_labels[sec + 1]:
                            next_action = temporal_labels[sec + 1][0]["action"]
                        
                        refined = self._refine_second_with_labelmap(
                            frames,
                            temporal_labels[sec],
                            labelmap,
                            prev_action,
                            next_action
                        )
                        refined_labels[sec] = refined
                    
                    result["temporal_labels"] = refined_labels
        
        # ==========================================
        # [KOR] Phase 4: 최종 id-time-action 결과 추출
        # [ENG] Phase 4: Extract final id-time-action results
        # ==========================================
        tube_results = {}
        id_time_actions = []
        
        for tube_id, result in tube_initial_results.items():
            temporal_labels = result.get("temporal_labels", {})
            final_actions = []
            
            for sec in sorted(temporal_labels.keys()):
                if temporal_labels[sec]:
                    top_candidate = temporal_labels[sec][0]
                    final_actions.append({
                        "time": sec,
                        "action": top_candidate["action"],
                        "confidence": top_candidate["confidence"]
                    })
                    id_time_actions.append({
                        "id": tube_id,
                        "time": sec,
                        "action": top_candidate["action"],
                        "confidence": top_candidate["confidence"]
                    })
            
            tube_results[tube_id] = {
                "tube_id": tube_id,
                "duration": result.get("duration", 0),
                "temporal_labels": {str(k): v for k, v in temporal_labels.items()},
                "final_actions": final_actions
            }
        
        # [KOR] id별, 그 다음 time별로 정렬
        # [ENG] Sort by id, then by time
        id_time_actions.sort(key=lambda x: (x["id"], x["time"]))
        
        # [KOR] 최종 결과 구성
        # [ENG] Build final result
        final_result = {
            "video_path": video_path,
            "labelmap": labelmap,
            "action_groups": action_groups,
            "tubes": tube_results,
            "id_time_actions": id_time_actions,
            "action_vocabulary": labelmap
        }
        
        # [KOR] 결과 저장
        # [ENG] Save results
        output_path = os.path.join(working_dir, "recognition_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f" │  ├─ Actions: {len(labelmap)} types in labelmap")
        print(f" │  └─ Temporal labels: {len(id_time_actions)} id-time-action entries")
        
        return final_result


def main():
    """
    [KOR] 인식기 테스트
    [ENG] Test the recognizer
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python recognizer.py <video_path> <tubes_dir>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    tubes_dir = sys.argv[2]
    
    recognizer = Recognizer()
    result = recognizer.recognize(video_path, tubes_dir)
    
    print("\n=== Recognition Results ===")
    print(f"Labelmap: {result.get('labelmap', [])}")
    print(f"\nAction groups:")
    for rep, originals in result.get("action_groups", {}).items():
        print(f"  {rep}: {originals}")
    print(f"\nID-Time-Action entries:")
    for entry in result.get("id_time_actions", [])[:20]:
        print(f"  {entry['id']} @ {entry['time']}s: {entry['action']} ({entry['confidence']:.2f})")
    
    if len(result.get("id_time_actions", [])) > 20:
        print(f"  ... and {len(result['id_time_actions']) - 20} more entries")


if __name__ == "__main__":
    main()