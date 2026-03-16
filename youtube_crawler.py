from collections import deque
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Crawling configuration
API_KEYS = []
CHANNEL_IDS = [
    "UCugbqfMO94F9guLEb6Olb2A",
    "UCF4Wxdo3inmxP-Y59wXDsFw",
    "UCHXvjavEtkPFJCfGlm0wTXw",
    "UCnHyx6H7fhKfUoAAaVGYvxQ",
    "UCWlV3Lz_55UaX4JsMj-z__Q",
    "UCH3mJ-nHxjjny2FJbJaqiDA",
]
KEYWORDS = ["대선", "윤석열", "이재명"]
START_DATE = "2022-03-01"
END_DATE = "2022-03-09"
TOTAL_LIMIT = 300_000
POLITICS_CATEGORY = "25"

# Output configuration
RAW_OUTPUT_CSV = "youtube_comments_full.csv"
FILTERED_OUTPUT_CSV = "youtube_filtered_comments_full.csv"
BOT_STATS_OUTPUT_CSV = "youtube_bot_analysis_stats_full.csv"


def _build_service_cache():
    return {}


def _rotate_call(fn, api_keys, exhausted, services, *args, **kwargs):
    for key in api_keys:
        if key in exhausted:
            continue
        try:
            if key not in services:
                services[key] = build("youtube", "v3", developerKey=key)
            return fn(services[key], *args, **kwargs)
        except HttpError as exc:
            if exc.resp.status == 403 and "quota" in str(exc).lower():
                exhausted.add(key)
                continue
            raise
    raise RuntimeError("All API keys are exhausted.")


def _search_videos(svc, channel_id, query, start_iso, end_iso, page_token=None, category=None):
    params = {
        "part": "id",
        "channelId": channel_id,
        "type": "video",
        "publishedAfter": start_iso + "Z",
        "publishedBefore": end_iso + "Z",
        "maxResults": 50,
        "pageToken": page_token,
        "fields": "nextPageToken,items(id/videoId)",
    }
    if query:
        params["q"] = query
    if category:
        params["videoCategoryId"] = category
    return svc.search().list(**params).execute()


def _fetch_video_info(svc, video_id):
    return svc.videos().list(
        part="snippet,statistics",
        id=video_id,
        fields=(
            "items(snippet(title,channelTitle,publishedAt,categoryId,tags),"
            "statistics(viewCount,likeCount,commentCount))"
        ),
    ).execute()


def _fetch_comments(svc, video_id, page_token=None):
    return svc.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        pageToken=page_token,
        textFormat="plainText",
        fields=(
            "nextPageToken,items/snippet/topLevelComment/snippet("
            "authorChannelId/value,textDisplay,publishedAt,likeCount)"
        ),
    ).execute()


def _fetch_channel_uploads(svc, channel_id, page_token=None):
    channel_resp = svc.channels().list(
        part="contentDetails",
        id=channel_id,
        fields="items/contentDetails/relatedPlaylists/uploads",
    ).execute()

    items = channel_resp.get("items", [])
    if not items:
        return {"items": [], "nextPageToken": None}

    upload_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    return svc.playlistItems().list(
        part="contentDetails",
        playlistId=upload_playlist_id,
        maxResults=50,
        pageToken=page_token,
        fields="nextPageToken,items/contentDetails/videoId",
    ).execute()


def crawl_youtube_comments(
    api_keys,
    channel_ids,
    keywords,
    start_date,
    end_date,
    total_limit,
    politics_category=POLITICS_CATEGORY,
):
    if not api_keys:
        raise ValueError("API_KEYS is empty. Please provide at least one YouTube API key.")

    exhausted = set()
    services = _build_service_cache()
    results = []
    counts = {channel_id: 0 for channel_id in channel_ids}
    per_channel_limit = total_limit // len(channel_ids)
    video_queues = {channel_id: deque() for channel_id in channel_ids}

    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    # Phase 1: Build video ID queues by date/channel/keyword/category.
    for day_offset in range((end_dt - start_dt).days):
        day_start = (start_dt + timedelta(days=day_offset)).isoformat()
        day_end = (start_dt + timedelta(days=day_offset + 1)).isoformat()

        for channel_id in channel_ids:
            for keyword in keywords:
                page_token = None
                while True:
                    response = _rotate_call(
                        _search_videos,
                        api_keys,
                        exhausted,
                        services,
                        channel_id,
                        keyword,
                        day_start,
                        day_end,
                        page_token,
                    )
                    for item in response.get("items", []):
                        video_queues[channel_id].append(item["id"]["videoId"])
                    page_token = response.get("nextPageToken")
                    if not page_token:
                        break

            page_token = None
            while True:
                response = _rotate_call(
                    _search_videos,
                    api_keys,
                    exhausted,
                    services,
                    channel_id,
                    None,
                    day_start,
                    day_end,
                    page_token,
                    category=politics_category,
                )
                for item in response.get("items", []):
                    video_queues[channel_id].append(item["id"]["videoId"])
                page_token = response.get("nextPageToken")
                if not page_token:
                    break

            if not video_queues[channel_id]:
                upload_token = None
                while True:
                    upload_response = _rotate_call(
                        _fetch_channel_uploads,
                        api_keys,
                        exhausted,
                        services,
                        channel_id,
                        upload_token,
                    )
                    for item in upload_response.get("items", []):
                        video_queues[channel_id].append(item["contentDetails"]["videoId"])
                    upload_token = upload_response.get("nextPageToken")
                    if not upload_token:
                        break

    # Phase 2: Round-robin comment collection by channel.
    done_all = False
    while sum(counts.values()) < total_limit and not done_all:
        done_all = True
        for channel_id in channel_ids:
            if counts[channel_id] >= per_channel_limit or not video_queues[channel_id]:
                continue

            done_all = False
            video_id = video_queues[channel_id].popleft()

            info_response = _rotate_call(
                _fetch_video_info,
                api_keys,
                exhausted,
                services,
                video_id,
            )
            items = info_response.get("items", [])
            if not items:
                continue
            info = items[0]

            base_row = {
                "video_id": video_id,
                "channel": info["snippet"]["channelTitle"],
                "video_title": info["snippet"]["title"],
                "video_published": info["snippet"]["publishedAt"],
                "video_views": info["statistics"].get("viewCount", 0),
                "video_likes": info["statistics"].get("likeCount", 0),
                "video_comment_count": info["statistics"].get("commentCount", 0),
                "video_category_id": info["snippet"]["categoryId"],
            }

            comment_token = None
            while counts[channel_id] < per_channel_limit:
                comments_response = _rotate_call(
                    _fetch_comments,
                    api_keys,
                    exhausted,
                    services,
                    video_id,
                    comment_token,
                )

                for thread in comments_response.get("items", []):
                    comment = thread["snippet"]["topLevelComment"]["snippet"]
                    author_channel = comment.get("authorChannelId", {})
                    results.append(
                        {
                            **base_row,
                            "comment_author_id": author_channel.get("value"),
                            "comment_text": comment.get("textDisplay", ""),
                            "comment_published": comment.get("publishedAt"),
                            "comment_likes": comment.get("likeCount", 0),
                        }
                    )
                    counts[channel_id] += 1
                    if counts[channel_id] >= per_channel_limit or sum(counts.values()) >= total_limit:
                        break

                comment_token = comments_response.get("nextPageToken")
                if not comment_token or counts[channel_id] >= per_channel_limit:
                    break

            time.sleep(0.1)

    return pd.DataFrame(results), counts


def optimized_filter_bot_comments(df):
    filtered_input = df.copy()
    filtered_input["ts"] = pd.to_datetime(filtered_input["comment_published"], errors="coerce")
    filtered_input.dropna(subset=["ts"], inplace=True)
    filtered_input.sort_values(["comment_author_id", "ts"], inplace=True)
    filtered_input["length"] = filtered_input["comment_text"].fillna("").str.len()
    filtered_input["repetition_ratio"] = (
        filtered_input["comment_text"]
        .fillna("")
        .str.split()
        .map(lambda words: 1 - len(set(words)) / len(words) if len(words) > 0 else 0)
    )
    filtered_input["interval"] = (
        filtered_input.groupby("comment_author_id")["ts"].diff().dt.total_seconds()
    )

    stats = filtered_input.groupby("comment_author_id").agg(
        comment_count=("comment_text", "size"),
        time_var_sec=(
            "interval",
            lambda values: np.var(values.dropna()) if len(values.dropna()) >= 2 else np.nan,
        ),
        length_var=("length", "var"),
        avg_rep_ratio=("repetition_ratio", "mean"),
    )
    stats["time_var_hr"] = stats["time_var_sec"] / 3600

    def ai_prob_series(texts):
        series = texts.fillna("").astype(str)
        sentence_lengths = series.str.split(".").map(
            lambda segments: [len(segment) for segment in segments if segment]
        )
        var_len = sentence_lengths.map(lambda lengths: np.var(lengths) if len(lengths) > 1 else 0)
        punct_count = series.str.count(r"[.!?]")
        sentence_count = series.str.count(r"\.") + 1
        punct_ratio = punct_count / sentence_count
        emotional = series.str.contains(r"ㅋㅋ|ㅎㅎ|ㅠㅠ|!!!|\?\?\?")
        return (
            (var_len < 50).astype(float)
            + (punct_ratio > 0.8).astype(float)
            + (~emotional & (series.str.len() > 50)).astype(float)
        ) / 3.0

    filtered_input["ai_prob"] = ai_prob_series(filtered_input["comment_text"])
    ai_stats = filtered_input.groupby("comment_author_id")["ai_prob"].mean().rename("avg_ai_prob")
    stats = stats.join(ai_stats, how="left").fillna(0)

    stats["cond_speed"] = ((stats["comment_count"] >= 5) & (stats["time_var_hr"] < 0.2)).astype(int)
    stats["cond_ai"] = (stats["avg_ai_prob"] > 0.5).astype(int)
    stats["suspicious_score"] = (
        (stats["length_var"] < 200).astype(float) * 0.2
        + (stats["avg_rep_ratio"] > 0.6).astype(float) * 0.3
    )
    stats["cond_suspicious"] = (stats["suspicious_score"] > 0.3).astype(int)
    stats["high_rep"] = (stats["avg_rep_ratio"] > 0.6).astype(int)
    stats["low_len_var"] = (stats["length_var"] < 200).astype(int)

    # Keep the stricter threshold from the old Youtube_crawling module.
    stats["is_bot"] = (
        stats[["cond_speed", "cond_ai", "cond_suspicious", "high_rep", "low_len_var"]].sum(axis=1) >= 3
    )

    bot_users = stats.index[stats["is_bot"]]
    filtered_df = filtered_input[~filtered_input["comment_author_id"].isin(bot_users)].drop(
        ["ts", "length", "repetition_ratio", "interval", "ai_prob"],
        axis=1,
    )
    return filtered_df.reset_index(drop=True), stats.reset_index()


def run_pipeline():
    comments_df, counts = crawl_youtube_comments(
        api_keys=API_KEYS,
        channel_ids=CHANNEL_IDS,
        keywords=KEYWORDS,
        start_date=START_DATE,
        end_date=END_DATE,
        total_limit=TOTAL_LIMIT,
    )
    comments_df.to_csv(RAW_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    filtered_df, stats_df = optimized_filter_bot_comments(comments_df)
    filtered_df.to_csv(FILTERED_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    stats_df.to_csv(BOT_STATS_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Channel counts:", counts)
    print("Raw comments:", len(comments_df))
    print("Filtered comments:", len(filtered_df))
    print("Saved:", RAW_OUTPUT_CSV, FILTERED_OUTPUT_CSV, BOT_STATS_OUTPUT_CSV)


if __name__ == "__main__":
    run_pipeline()
