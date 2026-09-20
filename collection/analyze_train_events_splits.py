"""Summarize how repeated ``(video id, label)`` events were split."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-events", type=Path, required=True)
    args = parser.parse_args()
    path = args.train_events
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    events = data["events"] if isinstance(data, dict) and "events" in data else data

    print("=== 基本统计 ===")
    print(f"总 event 数: {len(events)}")

    gm = Counter(e.get("group_method") for e in events)
    print(f"group_method 分布: {dict(gm)}")

    by_key: dict[tuple[str, str], list] = defaultdict(list)
    for e in events:
        key = (e["id"], e.get("label", ""))
        by_key[key].append(e)

    multi = {k: v for k, v in by_key.items() if len(v) > 1}
    print(f"唯一 (id, label) 组合数: {len(by_key)}")
    print(f"有多个 event 的 (id, label) 组合数: {len(multi)}")
    print(f"属于分割组的 event 总数: {sum(len(v) for v in multi.values())}")

    split_by_method: Counter = Counter()
    single_by_method: Counter = Counter()
    for v in by_key.values():
        if len(v) > 1:
            for e in v:
                split_by_method[e.get("group_method")] += 1
        else:
            single_by_method[v[0].get("group_method")] += 1

    print("\n=== 分割组内 group_method 分布 ===")
    print(dict(split_by_method))
    print("\n=== 单 event 组 group_method 分布 ===")
    print(dict(single_by_method))

    ruptures_splits = [
        (k, v)
        for k, v in multi.items()
        if any(e.get("group_method") == "ruptures" for e in v)
    ]
    temporal_splits = [
        (k, v)
        for k, v in multi.items()
        if all(e.get("group_method") == "temporal_gap" for e in v)
    ]
    mixed_splits = [
        (k, v)
        for k, v in multi.items()
        if len({e.get("group_method") for e in v}) > 1
    ]

    print("\n=== 分割类型 ===")
    print(f"含 ruptures 的分割组: {len(ruptures_splits)}")
    print(f"纯 temporal_gap 分割组: {len(temporal_splits)}")
    print(f"混合 group_method 分割组: {len(mixed_splits)}")

    if ruptures_splits:
        counts = [len(v) for _, v in ruptures_splits]
        cnt_dist = Counter(counts)
        print(
            f"ruptures 分割组 event 数: min={min(counts)}, max={max(counts)}, "
            f"avg={sum(counts) / len(counts):.2f}"
        )
        print(f"ruptures 分割组 event 数量分布: {dict(sorted(cnt_dist.items()))}")

    ei_gt0 = sum(1 for e in events if e.get("event_index", 0) > 0)
    print(f"\nevent_index > 0 的 event 数: {ei_gt0}")

    # ruptures 分割组内：是否全部用 ruptures
    all_ruptures_splits = [
        (k, v)
        for k, v in ruptures_splits
        if all(e.get("group_method") == "ruptures" for e in v)
    ]
    print(f"纯 ruptures 分割组 (组内全部 ruptures): {len(all_ruptures_splits)}")

    print("\n=== ruptures 分割示例 (event 数最多的前5组) ===")
    for i, (k, v) in enumerate(
        sorted(ruptures_splits, key=lambda x: -len(x[1]))[:5]
    ):
        vid, label = k
        v_sorted = sorted(v, key=lambda e: e["start_sec"])
        gaps = [
            v_sorted[j]["start_sec"] - v_sorted[j - 1]["end_sec"]
            for j in range(1, len(v_sorted))
        ]
        print(f"{i + 1}. id={vid}, label={label!r}, events={len(v)}")
        for e in v_sorted[:3]:
            print(
                f"   idx={e['event_index']} {e['start_sec']}-{e['end_sec']} "
                f"dur={e['duration_sec']} method={e['group_method']} clips={e['clip_count']}"
            )
        if len(v_sorted) > 3:
            print(f"   ... (+{len(v_sorted) - 3} more)")
        if gaps:
            print(
                f"   gaps(sec): min={min(gaps):.1f}, max={max(gaps):.1f}, "
                f"avg={sum(gaps) / len(gaps):.1f}"
            )

    print("\n=== temporal_gap 分割示例 (event 数最多的前3组) ===")
    for i, (k, v) in enumerate(
        sorted(temporal_splits, key=lambda x: -len(x[1]))[:3]
    ):
        vid, label = k
        v_sorted = sorted(v, key=lambda e: e["start_sec"])
        print(f"{i + 1}. id={vid}, label={label!r}, events={len(v)}")
        for e in v_sorted[:4]:
            print(
                f"   idx={e['event_index']} {e['start_sec']}-{e['end_sec']} "
                f"dur={e['duration_sec']}"
            )

    # 分析 ruptures 单 event vs 多 event 对比
    ruptures_events = [e for e in events if e.get("group_method") == "ruptures"]
    ruptures_single_keys = {
        k for k, v in by_key.items() if len(v) == 1 and v[0].get("group_method") == "ruptures"
    }
    ruptures_multi_keys = {
        k for k, v in multi.items() if any(e.get("group_method") == "ruptures" for e in v)
    }
    print("\n=== ruptures 方法细分 ===")
    print(f"ruptures event 总数: {len(ruptures_events)}")
    print(f"ruptures 单 event 的 (id,label) 数: {len(ruptures_single_keys)}")
    print(f"ruptures 多 event 的 (id,label) 数: {len(ruptures_multi_keys)}")
    ruptures_multi_event_count = sum(
        len(v) for k, v in multi.items() if k in ruptures_multi_keys
    )
    print(f"ruptures 分割组内 event 总数: {ruptures_multi_event_count}")
    print(
        f"ruptures 分割率 (多event组/全部ruptures组): "
        f"{len(ruptures_multi_keys) / (len(ruptures_single_keys) + len(ruptures_multi_keys)) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
