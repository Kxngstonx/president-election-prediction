import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from config import (
    data_path,
    ensure_data_dir,
    TOPIC_ASSIGNMENT_COLUMNS,
    extract_time_label_from_path,
    topic_assignment_output_path,
)

ensure_data_dir()
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models
from gensim.models import CoherenceModel

class CompleteLDASystem:
    def __init__(self, text_column='comment_text', num_topics=10):
        """
        완전한 LDA 토픽 모델링 시스템
        
        Parameters:
        text_column: 텍스트가 포함된 컬럼명
        num_topics: 추출할 토픽 수
        """
        self.text_column = text_column
        self.num_topics = num_topics
        self.vectorizer = None
        self.doc_term_matrix = None
        self.lda_model = None
        self.feature_names = None
        self.gensim_dictionary = None
        self.gensim_corpus = None
        self.gensim_lda_model = None
        self.tokenized_texts = None

        # 성능 평가 결과 저장
        self.evaluation_results = {}

    def load_and_preprocess_data(self, csv_files):
        """여러 CSV 파일을 로드하고 전처리"""
        all_texts = []
        file_info = []

        for csv_path in csv_files:
            try:
                print(f"데이터 로드 중: {csv_path}")
                df = pd.read_csv(csv_path, encoding='utf-8')

                texts = df[self.text_column].dropna().drop_duplicates().tolist()
                texts = [text.strip() for text in texts if text.strip()]

                print(f"{len(texts):,}개 유효 텍스트 추출")
                time_label = extract_time_label_from_path(csv_path)

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

        print(f"🔄 총 {len(all_texts):,}개 문서 로드 완료")
        return all_texts, file_info

    def preprocess_text(self, texts, min_df=2, max_df=0.95, max_features=1000):
        """텍스트 벡터화 및 Gensim 형식 변환"""
        print("텍스트 벡터화 중...")

        # 1. Scikit-learn CountVectorizer
        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            token_pattern=r'\b[가-힣]{2,}\b',
            lowercase=False
        )

        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"벡터화 완료: {self.doc_term_matrix.shape[0]:,}개 문서, {self.doc_term_matrix.shape[1]:,}개 단어")

        # 2. Gensim 형식으로 변환
        print("🔄 Gensim 형식 변환 중...")

        self.tokenized_texts = []
        for text in texts:
            tokens = [word for word in text.split() if len(word) >= 2]
            self.tokenized_texts.append(tokens)

        # 3. Gensim 사전 및 코퍼스 생성
        self.gensim_dictionary = corpora.Dictionary(self.tokenized_texts)
        self.gensim_dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=max_features)
        self.gensim_corpus = [self.gensim_dictionary.doc2bow(text) for text in self.tokenized_texts]

        print(f"📚 Gensim 사전 크기: {len(self.gensim_dictionary)}")

        return self.tokenized_texts

    def train_lda_model(self, max_iter=20, learning_method='online', random_state=42):
        """LDA 모델 학습 (Scikit-learn 및 Gensim 버전)"""
        print("LDA 모델 학습 중...")

        # 4. Scikit-learn LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            random_state=random_state,
            doc_topic_prior=1.0,
            topic_word_prior=0.1,
            n_jobs=-1
        )

        self.lda_model.fit(self.doc_term_matrix)

        # Gensim LDA (Coherence 계산용)
        self.gensim_lda_model = models.LdaModel(
            corpus=self.gensim_corpus,
            id2word=self.gensim_dictionary,
            num_topics=self.num_topics,
            random_state=random_state,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )

        print(f"LDA 모델 학습 완료: {self.num_topics}개 토픽")

    def calculate_coherence_score(self):
        """토픽 일관성 점수 계산"""
        print("Coherence 점수 계산 중...")

        try:
            coherence_model = CoherenceModel(
                model=self.gensim_lda_model,
                texts=self.tokenized_texts,
                dictionary=self.gensim_dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()

            print(f"Coherence Score (C_V): {coherence_score:.4f}")
            return coherence_score

        except Exception as e:
            print(f"Coherence 계산 실패: {e}")
            return 0.0

    def calculate_topic_diversity(self, top_k=25):
        """토픽 다양성 계산"""
        print("Topic Diversity 계산 중...")

        try:
            topic_words = []
            for topic_idx in range(self.num_topics):
                top_words_idx = self.lda_model.components_[topic_idx].argsort()[-top_k:][::-1]
                topic_words.append(set(top_words_idx))

            unique_words = set()
            total_words = 0

            for words in topic_words:
                unique_words.update(words)
                total_words += len(words)

            diversity_score = len(unique_words) / total_words if total_words > 0 else 0

            print(f"Topic Diversity: {diversity_score:.4f}")
            return diversity_score

        except Exception as e:
            print(f"Topic Diversity 계산 실패: {e}")
            return 0.0

    def calculate_perplexity(self):
        """Perplexity 점수 계산"""
        print("Perplexity 점수 계산 중...")

        try:
            # 5. 성능 점수 계산
            perplexity_score = self.lda_model.perplexity(self.doc_term_matrix)
            gensim_perplexity = self.gensim_lda_model.log_perplexity(self.gensim_corpus)

            print(f"Perplexity (Scikit-learn): {perplexity_score:.4f}")
            print(f"Log Perplexity (Gensim): {gensim_perplexity:.4f}")

            return perplexity_score, gensim_perplexity

        except Exception as e:
            print(f"Perplexity 계산 실패: {e}")
            return float('inf'), float('inf')

    def evaluate_model_performance(self):
        """종합적인 모델 성능 평가"""
        print("\n=== LDA 모델 성능 평가 ===")

        coherence_score = self.calculate_coherence_score()
        diversity_score = self.calculate_topic_diversity()
        perplexity_sklearn, perplexity_gensim = self.calculate_perplexity()

        self.evaluation_results = {
            'coherence_score': coherence_score,
            'topic_diversity': diversity_score,
            'perplexity_sklearn': perplexity_sklearn,
            'perplexity_gensim': perplexity_gensim,
            'num_topics': self.num_topics
        }

        print(f"\n=== 성능 평가 요약 ===")
        print(f"Coherence Score: {coherence_score:.4f}")
        print(f"Topic Diversity: {diversity_score:.4f}")
        print(f"Perplexity (Sklearn): {perplexity_sklearn:.4f}")
        print(f"Log Perplexity (Gensim): {perplexity_gensim:.4f}")

        return self.evaluation_results

    def display_topics(self, num_words=10):
        """토픽별 주요 단어 출력"""
        print("\n=== 토픽별 주요 단어 ===")

        topics_info = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-num_words:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]

            print(f"\n토픽 {topic_idx + 1}:")
            for word, weight in zip(top_words, top_weights):
                print(f"   {word}: {weight:.4f}")

            topics_info.append({
                'topic_id': topic_idx + 1,
                'words': top_words,
                'weights': top_weights
            })

        return topics_info

    def get_document_topics(self, texts):
        """각 문서의 토픽 분포 계산"""
        print("문서별 토픽 분포 계산 중...")

        doc_topic_probs = self.lda_model.transform(self.doc_term_matrix)

        document_topics = []
        for i, (text, doc_topics) in enumerate(zip(texts, doc_topic_probs)):
            dominant_topic = doc_topics.argmax()
            dominant_prob = doc_topics.max()

            document_topics.append({
                'document_id': i,
                'text': text,
                'dominant_topic': dominant_topic + 1,
                'probability': dominant_prob,
                'all_topic_probs': doc_topics.tolist()
            })

        return pd.DataFrame(document_topics)

    def export_topic_assignments(self, document_topics_df, file_info, topics_info, topic_method="lda"):
        """문서별 토픽 할당 결과를 표준 스키마 CSV로 저장."""
        if document_topics_df.empty:
            print("토픽 할당 데이터가 없어 저장을 건너뜁니다.")
            return []

        keyword_map = {
            int(topic["topic_id"]): ", ".join(topic["words"][:5])
            for topic in topics_info
        }

        rows = []
        for idx, row in document_topics_df.iterrows():
            meta = file_info[idx] if idx < len(file_info) else {}
            topic_id = int(row["dominant_topic"])
            rows.append({
                "comment_id": int(row["document_id"]),
                "time_label": meta.get("time_label", "unknown"),
                "comment_text": row["text"],
                "topic_id": topic_id,
                "topic_prob": float(row["probability"]),
                "topic_keywords": keyword_map.get(topic_id, f"topic_{topic_id}"),
                "topic_method": topic_method,
            })

        export_df = pd.DataFrame(rows, columns=TOPIC_ASSIGNMENT_COLUMNS)
        saved_paths = []
        for time_label, group in export_df.groupby("time_label"):
            out_path = topic_assignment_output_path(time_label, topic_method)
            group.to_csv(out_path, index=False, encoding="utf-8-sig")
            saved_paths.append(out_path)
            print(f"✅ 토픽 할당 저장: {out_path} ({len(group):,}건)")
        return saved_paths

def run_complete_lda_pipeline():
    """완전한 LDA 파이프라인 실행"""
    print("=== 완전한 LDA 토픽 모델링 파이프라인 시작 ===")

    csv_files = [
        data_path("comments_0204_0206.csv"),
        data_path("comments_0212_0214.csv"),
        data_path("comments_0222_0224.csv"),
        data_path("comments_0226_0228.csv"),
        data_path("comments_0303_0305.csv"),
    ]

    existing_files = [f for f in csv_files if os.path.exists(f)]
    print(f"처리 가능한 파일: {len(existing_files)}개")

    if not existing_files:
        print("처리할 CSV 파일이 없습니다.")
        return

    # LDA 시스템 객체 생성
    lda_system = CompleteLDASystem(
        text_column='comment_text',
        num_topics=10
    )

    # 1. 데이터 로드 및 전처리
    all_texts, file_info = lda_system.load_and_preprocess_data(existing_files)

    if not all_texts:
        print("유효한 텍스트 데이터가 없습니다.")
        return

    # 2. 텍스트 벡터화 및 코퍼스 생성
    tokenized_texts = lda_system.preprocess_text(
        all_texts,
        min_df=3,
        max_df=0.8,
        max_features=1000
    )

    # 3. LDA 모델 학습
    lda_system.train_lda_model(max_iter=20)

    # 4. 성능 평가
    evaluation_results = lda_system.evaluate_model_performance()

    # 5. 토픽 분석 및 결과 출력
    topics_info = lda_system.display_topics(num_words=10)

    # 6. 문서별 토픽 분포 계산
    document_topics_df = lda_system.get_document_topics(all_texts)

    print("\n=== 완전한 LDA 토픽 모델링 파이프라인 완료 ===")
    print(f"최종 성능 지표:")
    print(f"   Coherence: {evaluation_results['coherence_score']:.4f}")
    print(f"   Diversity: {evaluation_results['topic_diversity']:.4f}")
    print(f"   Perplexity: {evaluation_results['perplexity_sklearn']:.4f}")

    saved_topic_files = lda_system.export_topic_assignments(
        document_topics_df=document_topics_df,
        file_info=file_info,
        topics_info=topics_info,
        topic_method="lda",
    )

    return lda_system, evaluation_results, saved_topic_files

# 실행 코드
if __name__ == "__main__":
    try:
        import gensim
        print("Gensim 라이브러리 확인")
    except ImportError:
        print("Gensim 설치 필요")
        exit()

    lda_model, results, output_files = run_complete_lda_pipeline()
    print("생성된 토픽 할당 파일:", output_files)
