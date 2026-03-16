# -*- coding: utf-8 -*-
"""
파이프라인 공통 설정.
모든 입·출력 데이터는 data/ 디렉터리 기준으로 경로를 사용합니다.
데이터 계약(스키마·파일명·유틸)도 이 파일에서 관리합니다.
"""
import os
import re
from typing import Iterable, Sequence

# 프로젝트 루트 (이 파일이 있는 디렉터리)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 디렉터리 (입력/출력 CSV, 엑셀, 텍스트 등)
DATA_DIR = os.path.join(ROOT_DIR, "data")


def data_path(*parts):
    """data/ 아래 상대 경로를 절대 경로로 반환."""
    return os.path.join(DATA_DIR, *parts)


def ensure_data_dir():
    """data 디렉터리가 없으면 생성."""
    os.makedirs(DATA_DIR, exist_ok=True)


# ---------- 데이터 계약 (토픽/클러스터/XGBoost 입출력) ----------
DEFAULT_TOPIC_METHOD = "bertopic"
SUPPORTED_TOPIC_METHODS = {"bertopic", "lda", "etm"}

TOPIC_ASSIGNMENT_COLUMNS = [
    "comment_id",
    "time_label",
    "comment_text",
    "topic_id",
    "topic_prob",
    "topic_keywords",
    "topic_method",
]

CLUSTER_OUTPUT_COLUMNS = [
    "comment_id",
    "time_label",
    "comment_text",
    "topic_id",
    "topic_keywords",
    "candidate",
    "final_cluster",
]

XGBOOST_INPUT_COLUMNS = ["final_cluster", "negative", "positive", "time_label"]

DEFAULT_CANDIDATES = ["이재명", "김문수", "이준석"]


def extract_time_label_from_path(path: str) -> str:
    """파일명에서 time_label을 추출한다. 미검출 시 unknown."""
    name = os.path.basename(path)

    match = re.search(r"(\d{4}_\d{4})", name)
    if match:
        return match.group(1)

    match = re.search(r"(\d{4})_(\d{4})", name)
    if match:
        return f"{match.group(1)}_{match.group(2)}"

    match = re.search(r"(\d{8})~(\d{8})", name)
    if match:
        return f"{match.group(1)[4:]}_{match.group(2)[4:]}"

    return "unknown"


def topic_assignment_output_path(time_label: str, method: str) -> str:
    return data_path(f"topic_assignments_{time_label}_{method}.csv")


def missing_columns(columns: Sequence[str], required_columns: Iterable[str]):
    column_set = set(columns)
    return [col for col in required_columns if col not in column_set]
