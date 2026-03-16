import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from gensim.models import CoherenceModel
from tqdm import tqdm
import os

# GPU 환경 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

from config import (
    data_path,
    ensure_data_dir,
    TOPIC_ASSIGNMENT_COLUMNS,
    extract_time_label_from_path,
    topic_assignment_output_path,
)
ensure_data_dir()

class PoliticalDomainHyperParameters:
    """정치 도메인 특화 하이퍼파라미터"""

    # 모델 설정
    MODEL_NAME = 'monologg/koelectra-base-v3-discriminator'
    MAX_LENGTH = 128
    BATCH_SIZE = 32

    # 계층적 토픽 구조
    POLITICAL_HIERARCHY = {
        'level1': ['yoon', 'lee', 'neutral'],
        'level2': ['policy', 'scandal', 'campaign', 'general'],
        'level3_topics_per_person': 3
    }

    # ETM 파라미터
    NUM_TOPICS_TOTAL = 12
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 400
    DROPOUT = 0.5

    # 학습 파라미터
    EPOCHS = 100
    LEARNING_RATE = 0.001
    BETA1 = 0.99
    BETA2 = 0.999

    # 정치적 대립 정규화 파라미터
    OPPOSITION_WEIGHT = 2.0

    # 어휘 설정
    MIN_DF = 3
    MAX_DF = 0.8
    MAX_FEATURES = 1000

class KoELECTRAEmbedding:
    """KoELECTRA 임베딩 모델"""

    def __init__(self, model_name=PoliticalDomainHyperParameters.MODEL_NAME):
        print(f"KoELECTRA 모델 로드 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        print("KoELECTRA 모델 로드 완료")

    def encode_documents(self, texts, batch_size=PoliticalDomainHyperParameters.BATCH_SIZE):
        """문서 임베딩 생성"""
        embeddings = []
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="문서 임베딩 생성"):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=PoliticalDomainHyperParameters.MAX_LENGTH,
                    return_tensors='pt'
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def encode_words(self, words, batch_size=PoliticalDomainHyperParameters.BATCH_SIZE):
        """단어 임베딩 생성"""
        embeddings = []
        self.model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(words), batch_size), desc="단어 임베딩 생성"):
                batch = words[i:i+batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors='pt'
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

class PoliticalDocumentClassifier:
    """정치인별 문서 분류기"""

    def __init__(self):
        self.person_keywords = {
            'yoon': ['윤석열', '윤', '대통령', '현정부', '정부', '여당', '국민의힘'],
            'lee': ['이재명', '이', '민주당', '야당', '더불어민주당'],
            'neutral': []
        }

        self.issue_keywords = {
            'policy': ['정책', '공약', '법안', '제도', '개혁', '예산', '경제', '복지'],
            'scandal': ['수사', '검찰', '재판', '혐의', '비리', '논란', '의혹'],
            'campaign': ['선거', '캠페인', '유세', '공천', '후보', '지지율'],
            'general': ['발언', '회견', '성명', '입장', '반응', '비판']
        }

    def classify_documents(self, texts):
        """문서를 인물별, 이슈별로 분류"""
        person_labels = []
        issue_labels = []

        for text in tqdm(texts, desc="문서 분류 중"):
            text_lower = text.lower()

            # 인물 분류
            person_scores = {}
            for person, keywords in self.person_keywords.items():
                if person == 'neutral':
                    continue
                score = sum(text_lower.count(keyword) for keyword in keywords)
                person_scores[person] = score

            if max(person_scores.values()) > 0:
                dominant_person = max(person_scores, key=person_scores.get)
            else:
                dominant_person = 'neutral'

            person_labels.append(dominant_person)

            # 이슈 분류
            issue_scores = {}
            for issue, keywords in self.issue_keywords.items():
                score = sum(text_lower.count(keyword) for keyword in keywords)
                issue_scores[issue] = score

            if max(issue_scores.values()) > 0:
                dominant_issue = max(issue_scores, key=issue_scores.get)
            else:
                dominant_issue = 'general'

            issue_labels.append(dominant_issue)

        return person_labels, issue_labels

class HierarchicalPoliticalETM(nn.Module):
    """계층적 정치 도메인 특화 ETM"""

    def __init__(self, vocab_size, num_topics, embedding_dim, hidden_dim,
                 word_embeddings=None, topic_embeddings=None, dropout=0.5):
        super(HierarchicalPoliticalETM, self).__init__()

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 변분 인코더
        self.encoder_layers = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mu_layer = nn.Linear(hidden_dim, num_topics)
        self.logvar_layer = nn.Linear(hidden_dim, num_topics)

        # 단어 임베딩
        if word_embeddings is not None:
            if word_embeddings.shape[1] != embedding_dim:
                self.word_embedding_transform = nn.Linear(word_embeddings.shape[1], embedding_dim, bias=False)
                self.word_embeddings_raw = nn.Parameter(torch.FloatTensor(word_embeddings))
                self.word_embeddings_raw.requires_grad = False
            else:
                self.word_embeddings = nn.Parameter(torch.FloatTensor(word_embeddings))
                self.word_embeddings.requires_grad = False
                self.word_embedding_transform = None
        else:
            self.word_embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim))
            self.word_embedding_transform = None

        # 토픽 임베딩
        if topic_embeddings is not None:
            if topic_embeddings.shape[1] != embedding_dim:
                self.topic_embedding_transform = nn.Linear(topic_embeddings.shape[1], embedding_dim, bias=False)
                self.topic_embeddings_raw = nn.Parameter(torch.FloatTensor(topic_embeddings))
            else:
                self.topic_embeddings = nn.Parameter(torch.FloatTensor(topic_embeddings))
                self.topic_embedding_transform = None
        else:
            self.topic_embeddings = nn.Parameter(torch.randn(num_topics, embedding_dim))
            self.topic_embedding_transform = None

        # 배치 정규화
        self.batch_norm = nn.BatchNorm1d(num_topics, affine=False)

    def get_word_embeddings(self):
        """단어 임베딩 반환"""
        if self.word_embedding_transform is not None:
            return self.word_embedding_transform(self.word_embeddings_raw)
        else:
            return self.word_embeddings

    def get_topic_embeddings(self):
        """토픽 임베딩 반환"""
        if self.topic_embedding_transform is not None:
            return self.topic_embedding_transform(self.topic_embeddings_raw)
        else:
            return self.topic_embeddings

    def get_beta(self):
        """토픽-단어 분포 계산"""
        word_embeddings = self.get_word_embeddings()
        topic_embeddings = self.get_topic_embeddings()

        # 구면 공간에서의 방향성 유사도
        word_embeddings_norm = F.normalize(word_embeddings, p=2, dim=1)
        topic_embeddings_norm = F.normalize(topic_embeddings, p=2, dim=1)

        logits = torch.mm(topic_embeddings_norm, word_embeddings_norm.T)
        beta = F.softmax(logits, dim=1)

        return beta

    def forward(self, bow):
        """순전파"""
        # 정규화된 BoW
        normalized_bow = bow / (bow.sum(1, keepdim=True) + 1e-8)

        # 변분 추론
        h = self.encoder_layers(normalized_bow)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        # 재매개화
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            theta = mu + eps * std
        else:
            theta = mu

        # 배치 정규화
        if theta.size(0) > 1:
            theta = self.batch_norm(theta)

        # 소프트맥스로 토픽 분포 생성
        theta = F.softmax(theta, dim=1)

        # 토픽-단어 분포
        beta = self.get_beta()

        # 문서-단어 분포 재구성
        word_dist = torch.mm(theta, beta)

        return word_dist, theta, mu, logvar, beta

    def get_topics(self, vocab, top_k=10):
        """토픽별 상위 단어 추출"""
        beta = self.get_beta()
        topics = []

        for k in range(self.num_topics):
            topic_words = []
            _, top_indices = torch.topk(beta[k], top_k)
            for idx in top_indices:
                word = vocab[idx.item()]
                prob = beta[k][idx].item()
                topic_words.append((word, prob))
            topics.append(topic_words)

        return topics

class EnhancedHierarchicalETM:
    """향상된 계층적 ETM 클래스"""

    def __init__(self, text_column='comment_text', num_topics=12):
        self.text_column = text_column
        self.num_topics = num_topics

        # 모델 구성요소
        self.embedding_model = None
        self.classifier = None
        self.etm_model = None
        self.vectorizer = None
        self.vocab = None

        # 성능 평가 결과
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
                time_label = extract_time_label_from_path(csv_path)

                print(f"{len(texts):,}개 유효 텍스트 추출")

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

        print(f"총 {len(all_texts):,}개 문서 로드 완료")
        return all_texts, file_info

    def initialize_embeddings_and_topics(self, texts, person_labels, issue_labels):
        """임베딩 및 토픽 초기화"""
        print("임베딩 및 토픽 초기화 중...")

        # KoELECTRA 임베딩 모델 로드
        self.embedding_model = KoELECTRAEmbedding()

        # BoW 행렬 생성
        self.vectorizer = CountVectorizer(
            min_df=PoliticalDomainHyperParameters.MIN_DF,
            max_df=PoliticalDomainHyperParameters.MAX_DF,
            max_features=PoliticalDomainHyperParameters.MAX_FEATURES,
            token_pattern=r'\b[가-힣]{2,}\b'
        )

        bow_matrix = self.vectorizer.fit_transform(texts).toarray()
        self.vocab = self.vectorizer.get_feature_names_out()

        print(f"어휘 크기: {len(self.vocab)}")

        # 단어 임베딩 생성
        word_embeddings = self.embedding_model.encode_words(self.vocab.tolist())

        # 차원 조정
        if word_embeddings.shape[1] != PoliticalDomainHyperParameters.EMBEDDING_DIM:
            target_dim = min(PoliticalDomainHyperParameters.EMBEDDING_DIM, word_embeddings.shape[1])
            if target_dim > 0:
                pca = PCA(n_components=target_dim)
                word_embeddings = pca.fit_transform(word_embeddings)
                word_embeddings = normalize(word_embeddings, norm='l2')

        # 토픽 임베딩 초기화
        topic_embeddings = self._initialize_political_topic_embeddings(
            texts, person_labels, issue_labels
        )

        return bow_matrix, word_embeddings, topic_embeddings

    def _initialize_political_topic_embeddings(self, texts, person_labels, issue_labels):
        """정치인별 토픽 임베딩 초기화"""
        print("정치인별 토픽 임베딩 초기화 중...")

        doc_embeddings = self.embedding_model.encode_documents(texts)

        # 정치인별 평균 임베딩
        person_embeddings = {}
        hierarchy = PoliticalDomainHyperParameters.POLITICAL_HIERARCHY

        for person in hierarchy['level1']:
            if person == 'neutral':
                neutral_mask = np.array(person_labels) == 'neutral'
                if np.any(neutral_mask):
                    person_embeddings[person] = np.mean(doc_embeddings[neutral_mask], axis=0)
                else:
                    person_embeddings[person] = np.random.randn(doc_embeddings.shape[1])
            else:
                person_mask = np.array(person_labels) == person
                if np.any(person_mask):
                    person_embeddings[person] = np.mean(doc_embeddings[person_mask], axis=0)
                else:
                    person_embeddings[person] = np.random.randn(doc_embeddings.shape[1])

        # 이슈별 평균 임베딩
        issue_embeddings = {}
        for issue in hierarchy['level2']:
            issue_mask = np.array(issue_labels) == issue
            if np.any(issue_mask):
                issue_embeddings[issue] = np.mean(doc_embeddings[issue_mask], axis=0)
            else:
                issue_embeddings[issue] = np.random.randn(doc_embeddings.shape[1])

        # 계층적 토픽 임베딩 생성
        topic_embeddings = []
        for person in hierarchy['level1']:
            for issue in hierarchy['level2']:
                topic_emb = (person_embeddings[person] * 0.6 +
                           issue_embeddings[issue] * 0.4)
                topic_embeddings.append(topic_emb)

        return np.array(topic_embeddings)

    def train_etm_model(self, bow_matrix, word_embeddings, topic_embeddings):
        """ETM 모델 학습"""
        print("계층적 ETM 모델 학습 중...")

        # 모델 생성
        self.etm_model = HierarchicalPoliticalETM(
            vocab_size=len(self.vocab),
            num_topics=self.num_topics,
            embedding_dim=word_embeddings.shape[1],
            hidden_dim=PoliticalDomainHyperParameters.HIDDEN_DIM,
            word_embeddings=word_embeddings,
            topic_embeddings=topic_embeddings,
            dropout=PoliticalDomainHyperParameters.DROPOUT
        ).to(device)

        # 옵티마이저
        optimizer = Adam(
            self.etm_model.parameters(),
            lr=PoliticalDomainHyperParameters.LEARNING_RATE,
            betas=(PoliticalDomainHyperParameters.BETA1, PoliticalDomainHyperParameters.BETA2)
        )

        # 데이터 로더
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(bow_matrix))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=PoliticalDomainHyperParameters.BATCH_SIZE,
            shuffle=True,
            drop_last=False
        )

        # 학습
        for epoch in range(PoliticalDomainHyperParameters.EPOCHS):
            self.etm_model.train()
            total_loss = 0

            for batch in data_loader:
                bow = batch[0].to(device)
                optimizer.zero_grad()

                # 순전파
                word_dist, theta, mu, logvar, beta = self.etm_model(bow)

                # 손실 계산
                recon_loss = -torch.sum(bow * torch.log(word_dist + 1e-8), dim=1)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                loss = torch.mean(recon_loss + kl_div)

                # 역전파
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{PoliticalDomainHyperParameters.EPOCHS}, Loss: {total_loss/len(data_loader):.4f}")

        print("ETM 모델 학습 완료")

    def calculate_coherence_score(self, texts):
        """Coherence 점수 계산"""
        print("Coherence 점수 계산 중...")

        try:
            # 토큰화된 텍스트 생성
            tokenized_texts = []
            for text in texts:
                tokens = [word for word in text.split() if len(word) >= 2]
                tokenized_texts.append(tokens)

            # Gensim 사전 및 코퍼스 생성
            dictionary = corpora.Dictionary(tokenized_texts)
            dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=1000)
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

            # 토픽별 상위 단어 추출
            topics = self.etm_model.get_topics(self.vocab, top_k=10)
            topic_words = [[word for word, _ in topic] for topic in topics]

            # Coherence 모델 생성 및 점수 계산
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()

            print(f"Coherence Score: {coherence_score:.4f}")
            return coherence_score

        except Exception as e:
            print(f"Coherence 계산 실패: {e}")
            return 0.0

    def calculate_topic_diversity(self, top_k=25):
        """토픽 다양성 계산"""
        print("Topic Diversity 계산 중...")

        try:
            topics = self.etm_model.get_topics(self.vocab, top_k=top_k)

            # 각 토픽의 상위 k개 단어 집합
            topic_words = []
            for topic in topics:
                words = set([word for word, _ in topic])
                topic_words.append(words)

            # 토픽 간 고유 단어 비율 계산
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

    def calculate_perplexity(self, bow_matrix):
        """Perplexity 점수 계산"""
        print("Perplexity 점수 계산 중...")

        try:
            self.etm_model.eval()
            total_log_likelihood = 0
            total_words = 0

            with torch.no_grad():
                bow_tensor = torch.FloatTensor(bow_matrix).to(device)
                word_dist, _, _, _, _ = self.etm_model(bow_tensor)

                # 로그 우도 계산
                log_likelihood = torch.sum(bow_tensor * torch.log(word_dist + 1e-8))
                total_log_likelihood += log_likelihood.item()
                total_words += torch.sum(bow_tensor).item()

            # Perplexity 계산
            perplexity = np.exp(-total_log_likelihood / total_words)

            print(f"Perplexity: {perplexity:.4f}")
            return perplexity

        except Exception as e:
            print(f"Perplexity 계산 실패: {e}")
            return float('inf')

    def evaluate_model_performance(self, texts, bow_matrix):
        """종합적인 모델 성능 평가"""
        print("\n=== 계층적 ETM 모델 성능 평가 ===")

        # Coherence 점수
        coherence_score = self.calculate_coherence_score(texts)

        # Topic Diversity
        diversity_score = self.calculate_topic_diversity()

        # Perplexity
        perplexity_score = self.calculate_perplexity(bow_matrix)

        # 결과 저장
        self.evaluation_results = {
            'coherence_score': coherence_score,
            'topic_diversity': diversity_score,
            'perplexity': perplexity_score,
            'num_topics': self.num_topics
        }

        print(f"\n=== 성능 평가 요약 ===")
        print(f"Coherence Score: {coherence_score:.4f}")
        print(f"Topic Diversity: {diversity_score:.4f}")
        print(f"Perplexity: {perplexity_score:.4f}")

        return self.evaluation_results

    def display_topics(self, num_words=10):
        """토픽별 주요 단어 출력"""
        print("\n=== 계층적 토픽별 주요 단어 ===")

        topics = self.etm_model.get_topics(self.vocab, top_k=num_words)
        topics_info = []

        for topic_idx, topic_words in enumerate(topics):
            print(f"\n토픽 {topic_idx + 1}:")
            for word, weight in topic_words:
                print(f"   {word}: {weight:.4f}")

            topics_info.append({
                'topic_id': topic_idx + 1,
                'words': [word for word, _ in topic_words],
                'weights': [weight for _, weight in topic_words]
            })

        return topics_info

    def get_document_topics(self, bow_matrix, texts):
        """각 문서의 토픽 분포 계산"""
        print("문서별 토픽 분포 계산 중...")

        self.etm_model.eval()
        document_topics = []

        with torch.no_grad():
            bow_tensor = torch.FloatTensor(bow_matrix).to(device)
            _, theta, _, _, _ = self.etm_model(bow_tensor)

            for i, (text, doc_topics) in enumerate(zip(texts, theta.cpu().numpy())):
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

    def export_topic_assignments(self, document_topics_df, file_info, topics_info, topic_method="etm"):
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

def run_enhanced_hierarchical_etm_pipeline():
    """향상된 계층적 ETM 파이프라인 실행"""
    print("=== 향상된 계층적 ETM 토픽 모델링 파이프라인 시작 ===")

    # CSV 파일 리스트 정의 (data/ 기준)
    csv_files = [
        data_path("comments_0204_0206.csv"),
        data_path("comments_0212_0214.csv"),
        data_path("comments_0222_0224.csv"),
        data_path("comments_0226_0228.csv"),
        data_path("comments_0303_0305.csv"),
    ]

    # 존재하는 파일만 필터링
    existing_files = [f for f in csv_files if os.path.exists(f)]
    print(f"처리 가능한 파일: {len(existing_files)}개")

    if not existing_files:
        print("처리할 CSV 파일이 없습니다.")
        return

    # 계층적 ETM 모델링 객체 생성
    etm_analyzer = EnhancedHierarchicalETM(
        text_column='comment_text',
        num_topics=12
    )

    # 1. 데이터 로드 및 전처리
    all_texts, file_info = etm_analyzer.load_and_preprocess_data(existing_files)

    if not all_texts:
        print("유효한 텍스트 데이터가 없습니다.")
        return

    # 2. 정치인별 문서 분류
    classifier = PoliticalDocumentClassifier()
    person_labels, issue_labels = classifier.classify_documents(all_texts)

    print(f"인물별 분포: {dict(zip(*np.unique(person_labels, return_counts=True)))}")
    print(f"이슈별 분포: {dict(zip(*np.unique(issue_labels, return_counts=True)))}")

    # 3. 임베딩 및 토픽 초기화
    bow_matrix, word_embeddings, topic_embeddings = etm_analyzer.initialize_embeddings_and_topics(
        all_texts, person_labels, issue_labels
    )

    # 4. ETM 모델 학습
    etm_analyzer.train_etm_model(bow_matrix, word_embeddings, topic_embeddings)

    # 5. 성능 평가
    evaluation_results = etm_analyzer.evaluate_model_performance(all_texts, bow_matrix)

    # 6. 토픽 분석 및 결과 출력
    topics_info = etm_analyzer.display_topics(num_words=10)

    # 7. 문서별 토픽 분포 계산
    document_topics_df = etm_analyzer.get_document_topics(bow_matrix, all_texts)

    print("\n=== 계층적 ETM 토픽 모델링 파이프라인 완료 ===")
    print(f"최종 성능 지표:")
    print(f"   Coherence: {evaluation_results['coherence_score']:.4f}")
    print(f"   Diversity: {evaluation_results['topic_diversity']:.4f}")
    print(f"   Perplexity: {evaluation_results['perplexity']:.4f}")

    saved_topic_files = etm_analyzer.export_topic_assignments(
        document_topics_df=document_topics_df,
        file_info=file_info,
        topics_info=topics_info,
        topic_method="etm",
    )

    return etm_analyzer, evaluation_results, saved_topic_files

# 실행 코드
if __name__ == "__main__":
    # 필요한 라이브러리 설치 확인
    try:
        import gensim
        print("Gensim 라이브러리 확인")
    except ImportError:
        print("Gensim 설치 필요")
        exit()

    # 파이프라인 실행
    etm_model, results, output_files = run_enhanced_hierarchical_etm_pipeline()
    print("생성된 토픽 할당 파일:", output_files)
