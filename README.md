# president-election-prediction

네이버 뉴스 댓글과 유튜브 댓글을 수집하여 토픽 모델링·감성 분석을 거친 뒤, 후보자별 여론 데이터를 XGBoost로 학습해 대선 지지율/여론 흐름을 예측하는 파이프라인입니다.

## 파이프라인 구성

데이터는 `data/` 디렉터리를 기준으로 흐르며, 각 단계의 입출력 스키마는 [config.py](config.py)에서 공통으로 관리합니다.

```
1. 크롤링          naver_crawler.py / youtube_crawler.py
2. 토픽 모델링      BERTopic.py / LDA.py / ETM.py  (택 1)
3. 클러스터링       cluster.py
4. 감성 분석        sentiment.py  (steam_finetune.py로 파인튜닝한 모델 사용 가능)
5. 예측 모델링      XGBoost.py
```

### 1. 크롤링
- [naver_crawler.py](naver_crawler.py): 네이버 뉴스에서 키워드(후보자명, "대선") 기반 기사와 댓글을 Selenium/requests로 수집하고, 작성 패턴 기반 휴리스틱으로 봇 의심 댓글을 필터링합니다.
- [youtube_crawler.py](youtube_crawler.py): YouTube Data API로 지정 채널의 정치 관련 영상 댓글을 수집하며, API 키를 여러 개 등록해 쿼터 소진 시 자동으로 로테이션합니다. 봇 필터링 로직은 네이버 크롤러와 동일한 방식입니다.

두 스크립트 모두 상단의 설정 값(키워드, 기간, 채널 ID 등)을 수정한 뒤 `python naver_crawler.py` / `python youtube_crawler.py`로 직접 실행합니다.

### 2. 토픽 모델링
수집된 댓글(`comment_id, time_label, comment_text` 등)에 대해 아래 세 가지 방법론 중 하나로 토픽을 추출하고, `topic_assignments_{time_label}_{method}.csv`를 생성합니다.
- [BERTopic.py](BERTopic.py): SBERT 임베딩 기반 BERTopic
- [LDA.py](LDA.py): scikit-learn/gensim 기반 LDA
- [ETM.py](ETM.py): KoELECTRA 임베딩을 활용한 임베디드 토픽 모델(ETM)

### 3. 클러스터링
[cluster.py](cluster.py)는 토픽 할당 결과를 후보자(기본값: 이재명, 김문수, 이준석)와 매칭해 `(토픽, 후보자)` 단위의 `final_cluster`를 생성합니다. 댓글 임베딩과 후보자명 임베딩 간 코사인 유사도로 가장 가까운 후보자를 배정합니다.

```
python cluster.py --topic-method bertopic
```

### 4. 감성 분석
[sentiment.py](sentiment.py)는 KcELECTRA 기반 감성 분류 모델로 클러스터별 댓글의 긍정/부정/중립 비율을 계산해 XGBoost 입력용 요약 CSV(`cluster_sentiment_summary_*.csv`)를 생성합니다.

[steam_finetune.py](steam_finetune.py)는 Steam 리뷰 데이터(`data/steam.csv`)로 KcELECTRA를 파인튜닝하는 스크립트로, `sentiment.py`가 사용할 감성 분류 모델을 준비하는 용도입니다.

### 5. 예측 모델링
[XGBoost.py](XGBoost.py)는 클러스터별 긍정/부정 비율과 시간 구간(`time_label`)을 입력으로 받아 여론조사 발표 구간(기본 `DEFAULT_POLL_DATE_RANGES`)에 맞춰 XGBoost 및 Set-Transformer 스타일 피처 임베딩으로 지지율을 예측합니다.

## 데이터 계약
[config.py](config.py)에서 아래 항목들을 정의합니다.
- `TOPIC_ASSIGNMENT_COLUMNS`, `CLUSTER_OUTPUT_COLUMNS`, `XGBOOST_INPUT_COLUMNS`: 각 단계 입출력 필수 컬럼
- `DEFAULT_CANDIDATES`: 기본 후보자 목록
- `extract_time_label_from_path`: 파일명에서 시간 구간 라벨을 추출하는 유틸

## 요구 사항
Python 3.9+ 및 다음 라이브러리가 필요합니다: `pandas`, `numpy`, `selenium`, `beautifulsoup4`, `openpyxl`, `google-api-python-client`, `sentence-transformers`, `bertopic`, `gensim`, `scikit-learn`, `transformers`, `datasets`, `torch`, `xgboost`.

네이버 크롤러는 로컬 Chrome/ChromeDriver가, 유튜브 크롤러는 `youtube_crawler.py`의 `API_KEYS`에 YouTube Data API 키가 필요합니다.
