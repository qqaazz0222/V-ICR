#### **Step A. `recognizer.py` 구조 변경: "선(先) 발견, 후(後) 인식"**

현재 코드는 `recognize()` 함수 안에서 `_analyze_tube_temporal`(Phase 1)을 모든 튜브에 대해 먼저 실행합니다. 이는 시간이 매우 오래 걸립니다.

**수정 제안:**

1. **Phase 0 추가:** 전체 튜브 중 **움직임이 다른 대표 튜브 10~20개만 샘플링**합니다. (K-Means 활용)
2. **캡셔닝 & 라벨맵:** 샘플링된 튜브만 VLM에 넣어 자유 캡셔닝을 하고, 이를 LLM으로 요약하여 `label_map.txt`를 먼저 만듭니다.
3. **Phase 1 수행:** 확정된 `label_map`을 가지고 전체 튜브에 대해 **객관식 질문(Multiple Choice)**을 던집니다.

```python
# [수정안] modules/recognizer.py의 recognize 메서드 로직 변경

def recognize(self, ...):
    # 1. [NEW] Phase 0: Action Discovery (대표 행동 발견)
    if not predefined_labelmap:
        # (1) 전체 튜브에서 모션 특징(속도, 이동량 등) 추출
        # (2) K-Means로 대표 튜브 N개 샘플링
        # (3) 샘플링된 튜브에 대해 VLM Free-form Captioning
        # (4) 캡션들을 LLM으로 요약 -> label_map 생성
        labelmap = self._discover_action_vocabulary(metadata, tubes_dir)
    else:
        labelmap = predefined_labelmap

    # 2. Phase 1: Classification (확정된 라벨로 전체 분류)
    # 기존 _analyze_tube_temporal 함수가 'candidates'를 생성할 때
    # 처음부터 labelmap에 있는 행동들 중에서만 확률을 계산하도록 강제
    ...

```

#### **Step B. 정제(Refinement) 로직의 고도화**

현재 `_refine_second_with_labelmap`은 VLM에게 "앞뒤 상황이 이러니 다시 생각해봐"라고 텍스트로 묻는 방식입니다. 이를 **데이터(Feature) 기반**으로 바꿔야 합니다.

**수정 제안:**

1. **Feature 저장:** Phase 1 수행 시, VLM의 마지막 Layer 임베딩이나 CLIP Visual Feature를 저장합니다.
2. **군집화 (Clustering):** 저장된 Feature들을 `Scikit-learn`의 K-Means로 군집화합니다.
3. **라벨 보정:**
* 어떤 클립이 VLM은 '달리기'라고 했지만, Feature 공간 상에서는 '걷기' 군집의 한가운데에 있다면?
*  신뢰도를 낮추거나 '걷기'로 수정합니다.



```python
# [수정안] modules/recognizer.py 내부

def _refine_with_clustering(self, tube_results):
    # 1. 모든 세그먼트의 Feature 수집
    features = []
    for tube in tube_results:
        features.extend(tube['features'])
    
    # 2. 군집화 수행
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=len(labelmap))
    clusters = kmeans.fit_predict(features)
    
    # 3. 군집 중심과의 거리를 기반으로 라벨 재조정 (앙상블)
    # Final_Score = alpha * VLM_Score + (1-alpha) * Cluster_Score
    ...

```

---

### **3. 결론: 무엇부터 해야 할까요?**

지금 코드 상태가 매우 좋으므로, 전체를 뒤엎지 말고 **기능을 하나씩 "끼워 넣는" 방식**으로 진행하세요.

1. **가장 시급한 것 (Phase 0 구현):**
* `recognizer.py`에 `_discover_action_vocabulary()` 메서드를 추가하세요.
* 전체 영상을 다 돌리는 비효율을 막는 것이 최우선입니다.


2. **독창성 확보 (CoTracker 추가):**
* `detector.py`에서 튜브를 저장할 때, **CoTracker 시각화**가 포함되도록 수정하세요. 이것만 들어가도 논문의 그림(Figure)이 완전히 달라집니다.


3. **마지막 단계 (군집화 정제):**
* 위 두 가지가 돌아간 뒤에, 성능을 1~2% 더 짜내기 위해 `sklearn`을 활용한 정제 로직을 추가하세요.



현재 `run.py`나 `dataset.py` 등은 건드릴 필요 없이, `detector.py`와 `recognizer.py` 내부 로직만 강화하면 됩니다.