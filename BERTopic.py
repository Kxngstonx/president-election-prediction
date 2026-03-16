import os
import re
import pandas as pd
import numpy as np
from bertopic import BERTopic
from config import (
    data_path,
    ensure_data_dir,
    TOPIC_ASSIGNMENT_COLUMNS,
    extract_time_label_from_path,
    topic_assignment_output_path,
)

ensure_data_dir()
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

class UnifiedBERTopicModeling:
    def __init__(self, political_keywords, embedding_model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        """
        통합 BERTopic 모델링 클래스

        Parameters:
        political_keywords: 정치 키워드 리스트
        embedding_model_name: 임베딩 모델명
        """
        self.political_keywords = political_keywords
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.all_texts = []
        self.file_info = []
        self.embeddings = None
        self.evaluation_results = {}

    def load_all_csv_files(self, csv_files):
        """모든 CSV 파일을 로드하고 통합"""
        print("모든 CSV 파일 로드 및 통합 중...")

        all_texts = []
        file_info = []

        for csv_path in csv_files:
            if not os.path.exists(csv_path):
                print(f"파일 없음: {csv_path}")
                continue

            try:
                print(f"처리 중: {csv_path}")

                # 날짜/구간 라벨 추출
                time_label = extract_time_label_from_path(csv_path)

                # CSV 로드 및 텍스트 추출
                df = pd.read_csv(csv_path, encoding='utf-8')
                texts = df['comment_text'].dropna().drop_duplicates().tolist()
                texts = [text.strip() for text in texts if text.strip()]

                print(f"{csv_path}에서 {len(texts):,}개 유효 텍스트 추출")

                # 파일 정보 저장
                for text in texts:
                    file_info.append({
                        'file_path': csv_path,
                        'time_label': time_label,
                        'text': text
                    })

                all_texts.extend(texts)

            except Exception as e:
                print(f"파일 로드 실패 {csv_path}: {e}")
                continue

        self.all_texts = all_texts
        self.file_info = file_info

        print(f"총 {len(all_texts):,}개 문서 통합 완료")
        print(f"처리된 파일 수: {len(set([info['file_path'] for info in file_info]))}")

        return all_texts, file_info

    def _topic_keywords(self, topic_id, top_k=5):
        if topic_id == -1:
            return "outlier"
        topic_words = self.topic_model.get_topic(topic_id) or []
        return ", ".join([word for word, _ in topic_words[:top_k]])

    def _topic_probability(self, doc_idx, topic_id):
        if self.probs is None:
            return 0.0

        row = self.probs[doc_idx]
        if topic_id == -1:
            return float(np.max(row)) if len(row) > 0 else 0.0
        if topic_id < len(row):
            return float(row[topic_id])
        return float(np.max(row)) if len(row) > 0 else 0.0

    def export_topic_assignments(self, topic_method="bertopic"):
        """문서별 토픽 할당 결과를 time_label 단위 CSV로 저장."""
        if self.topics is None or not self.file_info:
            print("토픽 할당 결과가 없어 저장을 건너뜁니다.")
            return []

        rows = []
        for idx, (topic_id, info) in enumerate(zip(self.topics, self.file_info)):
            rows.append({
                "comment_id": idx,
                "time_label": info.get("time_label", "unknown"),
                "comment_text": info.get("text", ""),
                "topic_id": int(topic_id),
                "topic_prob": self._topic_probability(idx, int(topic_id)),
                "topic_keywords": self._topic_keywords(int(topic_id)),
                "topic_method": topic_method,
            })

        assignments_df = pd.DataFrame(rows, columns=TOPIC_ASSIGNMENT_COLUMNS)
        saved_paths = []

        for time_label, group in assignments_df.groupby("time_label"):
            out_path = topic_assignment_output_path(time_label, topic_method)
            group.to_csv(out_path, index=False, encoding="utf-8-sig")
            saved_paths.append(out_path)
            print(f"✅ 토픽 할당 저장: {out_path} ({len(group):,}건)")

        return saved_paths

    def train_unified_model(self):
        """통합된 데이터로 BERTopic 모델 학습"""
        if not self.all_texts:
            print("로드된 텍스트가 없습니다.")
            return

        print("통합 BERTopic 모델 학습 중...")
        print(f"학습 데이터: {len(self.all_texts):,}개 문서")

        # 임베딩 생성
        print("문서 임베딩 생성 중...")
        self.embeddings = self.embedding_model.encode(self.all_texts, show_progress_bar=True)

        # BERTopic 모델 설정 및 학습
        vectorizer_model = CountVectorizer(vocabulary=self.political_keywords)
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer_model,
            language="multilingual",
            calculate_probabilities=True,
            verbose=True
        )

        print("토픽 모델링 실행 중...")
        self.topics, self.probs = self.topic_model.fit_transform(self.all_texts, self.embeddings)

        # 토픽 정보 출력
        topics_info = self.topic_model.get_topic_info()
        valid_topics = topics_info[topics_info.Topic != -1]

        print(f"모델 학습 완료!")
        print(f"발견된 토픽 수: {len(valid_topics)}개")
        print(f"아웃라이어 문서 수: {sum(1 for t in self.topics if t == -1):,}개")

    def calculate_coherence_score(self):
        """통합 모델의 Coherence 점수 계산"""
        print("Coherence 점수 계산 중...")

        try:
            # BERTopic에서 토픽별 단어 추출
            topics_info = self.topic_model.get_topic_info()
            valid_topics = topics_info[topics_info.Topic != -1]

            if len(valid_topics) == 0:
                print("유효한 토픽이 없습니다.")
                return 0.0

            # 문서별 토픽 그룹화
            documents = pd.DataFrame({
                "Document": self.all_texts,
                "ID": range(len(self.all_texts)),
                "Topic": self.topics
            })
            documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

            # BERTopic 전처리 적용
            cleaned_docs = self.topic_model._preprocess_text(documents_per_topic.Document.values)

            # 벡터라이저와 분석기 추출
            vectorizer = self.topic_model.vectorizer_model
            analyzer = vectorizer.build_analyzer()

            # 토큰화
            tokens = [analyzer(doc) for doc in cleaned_docs]
            dictionary = corpora.Dictionary(tokens)
            corpus = [dictionary.doc2bow(token) for token in tokens]

            # 토픽별 단어 추출 (유효한 단어만)
            topic_words = []
            for topic_id in valid_topics['Topic'].tolist():
                words = [word for word, _ in self.topic_model.get_topic(topic_id)]
                words = [word for word in words if word in dictionary.token2id]
                if len(words) > 0:
                    topic_words.append(words)

            if len(topic_words) == 0:
                print("유효한 토픽 단어가 없습니다.")
                return 0.0

            # Coherence 모델 생성 및 점수 계산
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=tokens,
                corpus=corpus,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()

            print(f"Coherence Score (C_V): {coherence_score:.4f}")
            return coherence_score

        except Exception as e:
            print(f"Coherence 계산 실패: {e}")
            return 0.0

    def calculate_topic_diversity(self, top_k=25):
        """통합 모델의 토픽 다양성 계산"""
        print("Topic Diversity 계산 중...")

        try:
            topics_info = self.topic_model.get_topic_info()
            valid_topics = topics_info[topics_info.Topic != -1]

            if len(valid_topics) == 0:
                print("유효한 토픽이 없습니다.")
                return 0.0

            # 각 토픽의 상위 k개 단어 추출
            all_topic_words = set()
            total_words = 0

            for topic_id in valid_topics['Topic'].tolist():
                topic_words = [word for word, _ in self.topic_model.get_topic(topic_id)[:top_k]]
                all_topic_words.update(topic_words)
                total_words += len(topic_words)

            # 토픽 간 고유 단어 비율 계산
            diversity_score = len(all_topic_words) / total_words if total_words > 0 else 0

            print(f"Topic Diversity: {diversity_score:.4f}")
            return diversity_score

        except Exception as e:
            print(f"Topic Diversity 계산 실패: {e}")
            return 0.0

    def calculate_perplexity(self):
        """통합 모델의 Perplexity 점수 계산"""
        print("Perplexity 점수 계산 중...")

        try:
            if self.probs is None:
                print("확률 정보가 없습니다.")
                return float('inf')

            # 유효한 확률 값만 필터링 (0이 아닌 값)
            valid_probs = [prob for prob in self.probs if prob > 0]

            if len(valid_probs) == 0:
                print("유효한 확률 값이 없습니다.")
                return float('inf')

            # Log perplexity 계산
            log_perplexity = -np.mean([np.log(prob) for prob in valid_probs])

            # Perplexity 계산
            perplexity = np.exp(log_perplexity)

            print(f"Perplexity: {perplexity:.4f}")
            return perplexity

        except Exception as e:
            print(f"Perplexity 계산 실패: {e}")
            return float('inf')

    def evaluate_unified_model_performance(self):
        """통합 모델의 종합적인 성능 평가"""
        print("\n=== 통합 BERTopic 모델 성능 평가 ===")

        if self.topic_model is None:
            print("학습된 모델이 없습니다.")
            return {}

        # 기본 정보
        topics_info = self.topic_model.get_topic_info()
        valid_topics = topics_info[topics_info.Topic != -1]
        num_topics = len(valid_topics)
        num_documents = len(self.all_texts)
        num_files = len(set([info['file_path'] for info in self.file_info]))

        print(f"통합 데이터셋 정보:")
        print(f"   처리된 파일 수: {num_files}")
        print(f"   총 문서 수: {num_documents:,}")
        print(f"   발견된 토픽 수: {num_topics}")

        # 성능 지표 계산
        coherence_score = self.calculate_coherence_score()
        diversity_score = self.calculate_topic_diversity()
        perplexity_score = self.calculate_perplexity()

        # 결과 저장
        self.evaluation_results = {
            'coherence_score': coherence_score,
            'topic_diversity': diversity_score,
            'perplexity': perplexity_score,
            'num_topics': num_topics,
            'num_documents': num_documents,
            'num_files': num_files,
            'dataset_type': 'unified_all_files'
        }

        print(f"\n=== 통합 모델 성능 평가 요약 ===")
        print(f"Coherence Score: {coherence_score:.4f}")
        print(f"Topic Diversity: {diversity_score:.4f}")
        print(f"Perplexity: {perplexity_score:.4f}")

        return self.evaluation_results

    def display_unified_topics(self, num_topics=15):
        """통합 모델의 토픽별 주요 키워드 출력"""
        print(f"\n=== 통합 모델 상위 {num_topics}개 토픽 키워드 ===")

        topics_info = self.topic_model.get_topic_info()
        top_topics = topics_info[topics_info.Topic != -1].head(num_topics)

        topic_details = []
        for _, row in top_topics.iterrows():
            topic_id = row["Topic"]
            keywords = self.topic_model.get_topic(topic_id)

            if keywords:
                print(f"\n토픽 {topic_id} (문서 수: {row['Count']:,}):")
                topic_words = []
                topic_weights = []

                for word, weight in keywords[:10]:
                    print(f"   {word}: {weight:.4f}")
                    topic_words.append(word)
                    topic_weights.append(weight)

                topic_details.append({
                    'topic_id': topic_id,
                    'document_count': row['Count'],
                    'words': topic_words,
                    'weights': topic_weights
                })

        return topic_details

    def get_file_distribution_analysis(self):
        """파일별 토픽 분포 분석"""
        print("\n=== 파일별 토픽 분포 분석 ===")

        # 파일별 토픽 분포 계산
        file_topic_dist = {}

        for i, (topic, info) in enumerate(zip(self.topics, self.file_info)):
            file_path = info['file_path']
            date_range = info.get('time_label', 'unknown')

            if date_range not in file_topic_dist:
                file_topic_dist[date_range] = {'total_docs': 0, 'topics': {}}

            file_topic_dist[date_range]['total_docs'] += 1

            if topic not in file_topic_dist[date_range]['topics']:
                file_topic_dist[date_range]['topics'][topic] = 0
            file_topic_dist[date_range]['topics'][topic] += 1

        # 결과 출력
        for date_range, dist_info in file_topic_dist.items():
            print(f"\n{date_range}:")
            print(f"   총 문서 수: {dist_info['total_docs']:,}")

            # 상위 5개 토픽 출력
            top_topics = sorted(dist_info['topics'].items(),
                              key=lambda x: x[1], reverse=True)[:5]

            for topic_id, count in top_topics:
                if topic_id != -1:  # 아웃라이어 제외
                    percentage = (count / dist_info['total_docs']) * 100
                    print(f"   토픽 {topic_id}: {count:,}개 ({percentage:.1f}%)")

        return file_topic_dist

def run_unified_bertopic_pipeline():
    """통합 BERTopic 파이프라인 실행"""
    print("=== 통합 BERTopic 토픽 모델링 파이프라인 시작 ===")

    # 1. 2025 대선 이슈 기반 키워드 사전 정의
    political_keywords_2025 = [
        # 주요 후보자
        "이재명", "김문수", "이준석", "권영국", "한덕수",

        # 정당 및 정치세력
        "더불어민주당", "국민의힘", "개혁신당", "민주노동당", "무소속", "비례정당", "탄핵"

        # 경제 및 복지
        "부동산", "전세", "청년취업", "기본소득", "노동개혁", "최저임금", "재정적자", "육아휴직", "청년정책",

        # 디지털 및 AI
        "AI", "인공지능", "디지털전환", "플랫폼노동", "데이터주권", "사이버보안",

        # 외교 및 안보
        "북한", "한미동맹", "한중관계", "전작권", "핵무장", "사드", "안보",

        # 사법 및 개혁
        "검찰개혁", "사법개혁", "공수처", "검수완박", "특검",

        # 선거 및 정치제도
        "사전투표", "부정선거", "여론조사", "TV토론", "후보단일화", "정권교체"
    ]

    # 3. 처리할 CSV 리스트 (data/ 기준)
    csv_files = [
        data_path("2025_comment_0519_0521.csv"),
        data_path("2025_comment_0524_0526.csv"),
        data_path("2025_comment_0528_0530.csv"),
    ]

    # 3. 통합 BERTopic 모델링 객체 생성
    unified_bertopic = UnifiedBERTopicModeling(political_keywords_2025)

    # 4. 모든 파일 로드 및 통합
    all_texts, file_info = unified_bertopic.load_all_csv_files(csv_files)

    if not all_texts:
        print("처리할 텍스트가 없습니다.")
        return None, None

    # 5. 통합 모델 학습
    unified_bertopic.train_unified_model()

    # 6. 통합 성능 평가
    evaluation_results = unified_bertopic.evaluate_unified_model_performance()

    # 7. 통합 토픽 정보 출력
    topic_details = unified_bertopic.display_unified_topics(num_topics=15)

    # 8. 파일별 분포 분석
    file_distribution = unified_bertopic.get_file_distribution_analysis()

    print("\n=== 통합 BERTopic 파이프라인 완료 ===")
    print(f"최종 통합 성능 지표:")
    print(f"   처리된 파일 수: {evaluation_results['num_files']}")
    print(f"   총 문서 수: {evaluation_results['num_documents']:,}")
    print(f"   발견된 토픽 수: {evaluation_results['num_topics']}")
    print(f"   Coherence: {evaluation_results['coherence_score']:.4f}")
    print(f"   Diversity: {evaluation_results['topic_diversity']:.4f}")
    print(f"   Perplexity: {evaluation_results['perplexity']:.4f}")

    # 9. 문서별 토픽 할당 CSV 저장
    saved_topic_files = unified_bertopic.export_topic_assignments(topic_method="bertopic")

    return unified_bertopic, evaluation_results, saved_topic_files

# 실행 코드
if __name__ == "__main__":
    # 파이프라인 실행
    model, results, output_files = run_unified_bertopic_pipeline()
    print("생성된 토픽 할당 파일:", output_files)