import argparse
import glob
import os

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

from config import data_path, ensure_data_dir, XGBOOST_INPUT_COLUMNS, missing_columns

ensure_data_dir()


def resolve_model_candidates(model_path=None):
    candidates = []
    if model_path:
        candidates.append(model_path)

    local_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "kcelectra_sentiment_final"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "steam_kcelectra"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    ]
    candidates.extend(local_candidates)
    candidates.append("beomi/KcELECTRA-base")
    return candidates


def build_pipeline(model_path=None):
    device = 0 if torch.cuda.is_available() else -1
    print("사용 디바이스:", "GPU" if device == 0 else "CPU")

    last_error = None
    for candidate in resolve_model_candidates(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            model = AutoModelForSequenceClassification.from_pretrained(candidate)
            pipe = TextClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                device=device,
                truncation=True,
                max_length=128,
            )
            print(f"✅ 감성 모델 로드: {candidate}")
            return pipe, candidate
        except Exception as err:
            last_error = err
            print(f"모델 로드 실패, 다음 후보로 진행: {candidate}")

    raise RuntimeError(f"사용 가능한 KcELECTRA 감성 모델을 찾지 못했습니다: {last_error}")


def map_label_to_sentiment(label, id2label, num_labels):
    label_str = str(label).strip().lower()

    if any(key in label_str for key in ["positive", "pos", "긍정"]):
        return "positive"
    if any(key in label_str for key in ["negative", "neg", "부정"]):
        return "negative"
    if any(key in label_str for key in ["neutral", "none", "중립"]):
        return "neutral"
    if any(key in label_str for key in ["hate", "offensive"]):
        return "negative"

    if label_str.startswith("label_"):
        try:
            label_idx = int(label_str.split("_")[1])
            label_name = str(id2label.get(label_idx, "")).lower()
            if any(key in label_name for key in ["positive", "pos", "긍정"]):
                return "positive"
            if any(key in label_name for key in ["negative", "neg", "부정", "hate", "offensive"]):
                return "negative"
            if any(key in label_name for key in ["neutral", "none", "중립"]):
                return "neutral"
            if num_labels == 2:
                return "positive" if label_idx == 1 else "negative"
        except Exception:
            pass

    return "neutral"


def load_clustered_comments(input_files=None):
    files = sorted(input_files) if input_files else sorted(glob.glob(data_path("clustered_*.csv")))
    if not files:
        raise FileNotFoundError("입력 clustered_*.csv 파일이 없습니다.")

    frames = []
    for path in files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        required = ["final_cluster", "comment_text", "time_label"]
        missing = missing_columns(df.columns, required)
        if missing:
            raise ValueError(f"{path} 필수 컬럼 누락: {missing}")
        frames.append(df[["final_cluster", "comment_text", "time_label"]].copy())

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["comment_text", "final_cluster", "time_label"]).copy()
    merged["comment_text"] = merged["comment_text"].astype(str)
    return merged


def build_cluster_sentiment_summary(
    model_path=None,
    input_files=None,
    output_path=None,
    detail_output_path=None,
    batch_size=32,
):
    df = load_clustered_comments(input_files=input_files)
    classifier, loaded_model_path = build_pipeline(model_path=model_path)

    predictions = classifier(
        df["comment_text"].tolist(),
        batch_size=batch_size,
        truncation=True,
        max_length=128,
    )
    id2label = classifier.model.config.id2label or {}
    num_labels = int(getattr(classifier.model.config, "num_labels", 2))

    df["sentiment_class"] = [
        map_label_to_sentiment(pred["label"], id2label, num_labels)
        for pred in predictions
    ]

    summary_rows = []
    grouped = df.groupby(["time_label", "final_cluster"], dropna=False)
    for (time_label, final_cluster), group in grouped:
        total = len(group)
        pos = (group["sentiment_class"] == "positive").sum()
        neg = (group["sentiment_class"] == "negative").sum()
        neu = (group["sentiment_class"] == "neutral").sum()

        summary_rows.append({
            "final_cluster": final_cluster,
            "positive": round(pos / total, 6) if total else 0.0,
            "negative": round(neg / total, 6) if total else 0.0,
            "neutral": round(neu / total, 6) if total else 0.0,
            "total_comments": total,
            "time_label": time_label,
            "model_path": loaded_model_path,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["time_label", "final_cluster"])
    required = missing_columns(summary_df.columns, XGBOOST_INPUT_COLUMNS)
    if required:
        raise ValueError(f"감성 요약 결과 필수 컬럼 누락: {required}")

    output_path = output_path or data_path("cluster_sentiment_summary_final_21대.csv")
    detail_output_path = detail_output_path or data_path("cluster_comment_sentiment_predictions.csv")

    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    df.to_csv(detail_output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 감성 요약 저장: {output_path}")
    print(f"✅ 댓글 단위 감성 저장: {detail_output_path}")
    return summary_df, output_path


def main():
    parser = argparse.ArgumentParser(description="클러스터 댓글 감성 추론(KcELECTRA) 및 XGBoost 입력 생성")
    parser.add_argument("--model-path", default=None, help="파인튜닝된 KcELECTRA 체크포인트 경로")
    parser.add_argument("--input-files", nargs="*", default=None, help="clustered CSV 파일 목록")
    parser.add_argument("--output-path", default=None, help="요약 CSV 출력 경로")
    parser.add_argument("--detail-output-path", default=None, help="댓글 단위 결과 출력 경로")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    build_cluster_sentiment_summary(
        model_path=args.model_path,
        input_files=args.input_files,
        output_path=args.output_path,
        detail_output_path=args.detail_output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()