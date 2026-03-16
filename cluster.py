import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    data_path,
    ensure_data_dir,
    CLUSTER_OUTPUT_COLUMNS,
    DEFAULT_CANDIDATES,
    DEFAULT_TOPIC_METHOD,
    SUPPORTED_TOPIC_METHODS,
    TOPIC_ASSIGNMENT_COLUMNS,
    extract_time_label_from_path,
    missing_columns,
)

ensure_data_dir()


def build_encoder():
    hf_model = "beomi/KcELECTRA-base"
    word_embedding_model = models.Transformer(hf_model, max_seq_length=128)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def get_topic_files(topic_method, input_pattern=None, explicit_files=None):
    if explicit_files:
        return sorted(explicit_files)
    if input_pattern:
        return sorted(glob.glob(input_pattern))
    pattern = data_path(f"topic_assignments_*_{topic_method}.csv")
    return sorted(glob.glob(pattern))


def normalize_topic_label(row):
    topic_id = row.get("topic_id", "unknown")
    keywords = str(row.get("topic_keywords", "")).strip()
    if not keywords or keywords.lower() == "nan":
        return f"topic_{topic_id}"
    first_keyword = keywords.split(",")[0].strip()
    clean = re.sub(r"\s+", "_", first_keyword)
    return clean if clean else f"topic_{topic_id}"


def cluster_topic_pairs(topic_file, model, candidate_embs, candidates, batch_size=64):
    df = pd.read_csv(topic_file, encoding="utf-8-sig")
    required = TOPIC_ASSIGNMENT_COLUMNS[:-1] + ["topic_method"]
    missing = missing_columns(df.columns, required)
    if missing:
        raise ValueError(f"{topic_file} 필수 컬럼 누락: {missing}")

    df = df.dropna(subset=["comment_text"]).copy()
    if df.empty:
        print(f"⚠️ 텍스트가 없어 건너뜀: {topic_file}")
        return None

    comments = df["comment_text"].astype(str).tolist()
    comment_embs = model.encode(
        comments,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    sims = cosine_similarity(comment_embs, candidate_embs)
    best_idx = np.argmax(sims, axis=1)
    df["candidate"] = [candidates[idx] for idx in best_idx]

    df["topic_label"] = df.apply(normalize_topic_label, axis=1)
    df["final_cluster"] = df["topic_label"] + "_" + df["candidate"]

    time_label = (
        str(df["time_label"].iloc[0])
        if "time_label" in df.columns and not df["time_label"].isna().all()
        else extract_time_label_from_path(topic_file)
    )
    output_path = data_path(f"clustered_{time_label}.csv")

    out_df = df.copy()
    for col in CLUSTER_OUTPUT_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan

    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {output_path} ({len(out_df):,}건)")
    print(out_df["final_cluster"].value_counts().head(10))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="토픽 할당 결과를 (토픽, 후보자) 페어로 클러스터링")
    parser.add_argument(
        "--topic-method",
        default=DEFAULT_TOPIC_METHOD,
        choices=sorted(SUPPORTED_TOPIC_METHODS),
        help="입력 토픽 파일의 방법론",
    )
    parser.add_argument(
        "--input-pattern",
        default=None,
        help="토픽 할당 CSV glob 패턴 (예: /path/topic_assignments_*_bertopic.csv)",
    )
    parser.add_argument(
        "--topic-files",
        nargs="*",
        default=None,
        help="처리할 토픽 할당 파일 경로 목록",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    topic_files = get_topic_files(
        topic_method=args.topic_method,
        input_pattern=args.input_pattern,
        explicit_files=args.topic_files,
    )
    if not topic_files:
        print("❌ 처리할 topic_assignments 파일이 없습니다.")
        return

    model = build_encoder()
    candidates = DEFAULT_CANDIDATES
    candidate_embs = model.encode(candidates, convert_to_numpy=True)

    for topic_file in topic_files:
        print(f"\n📂 Processing: {topic_file}")
        cluster_topic_pairs(
            topic_file=topic_file,
            model=model,
            candidate_embs=candidate_embs,
            candidates=candidates,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()