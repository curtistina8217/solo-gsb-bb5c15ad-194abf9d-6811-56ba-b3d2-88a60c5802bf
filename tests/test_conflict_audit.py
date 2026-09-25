"""冲突车流安全间隔审计核心算法测试。

确定性用例覆盖：同时绿灯、绿灯结束不足清空间隔的双向危险、清空间隔端点
（恰为 d 秒不危险）、两车尾段重叠不危险、审计范围起点之前的绿灯尾部伸入
范围、跨零点绿灯、常绿、offset 相位、不同周期、范围裁剪、d=0 与多对车流。
另含 1500 组随机参数下与逐秒判定暴力参考实现的逐项对照。
"""

import random

from app.greenwave import compute_conflict_audit, is_green


def brute_force(streams, pairs, audit_from, audit_to):
    """逐秒判定的参考实现：完全按题面三条危险规则。"""

    def danger(first, second, clearance, now):
        green_first = is_green(
            streams[first]["cycle"], streams[first]["offset"],
            streams[first]["green_start"], streams[first]["green_end"], now,
        )
        green_second = is_green(
            streams[second]["cycle"], streams[second]["offset"],
            streams[second]["green_start"], streams[second]["green_end"], now,
        )
        if green_first and green_second:
            return True
        if clearance and green_first:
            for back in range(1, clearance + 1):
                if is_green(
                    streams[second]["cycle"], streams[second]["offset"],
                    streams[second]["green_start"],
                    streams[second]["green_end"], now - back,
                ):
                    return True
        if clearance and green_second:
            for back in range(1, clearance + 1):
                if is_green(
                    streams[first]["cycle"], streams[first]["offset"],
                    streams[first]["green_start"],
                    streams[first]["green_end"], now - back,
                ):
                    return True
        return False

    results = []
    for first, second, clearance in pairs:
        seconds = [
            now for now in range(audit_from, audit_to)
            if danger(first, second, clearance, now)
        ]
        intervals = []
        for now in seconds:
            if intervals and intervals[-1][1] == now:
                intervals[-1][1] = now + 1
            else:
                intervals.append([now, now + 1])
        results.append(intervals)
    return results


def stream(green_start=0, green_end=30, cycle=60, offset=0):
    return {
        "cycle": cycle,
        "offset": offset,
        "green_start": green_start,
        "green_end": green_end,
    }


# ---------- 确定性用例 ----------


def test_simultaneous_green_is_dangerous():
    streams = [stream(0, 30), stream(10, 40)]
    # d=0 时危险集合恰为绿灯交集 [10,30)（每股周期重复）
    assert compute_conflict_audit(streams, [(0, 1, 0)], 0, 120) == [
        [[10, 30], [70, 90]]
    ]


def test_clearance_tail_after_one_green_ends():
    # 两股绿灯首尾相接：s0 [0,30) 与 s1 [30,60)，d=5
    # t∈[0,5)：s0 绿，而 s1 上周期绿灯 [-30,0) 在 t=0 刚结束（不足 5 秒）；
    # t∈[30,35)：s1 绿，而 s0 在 30 刚结束。两段均危险，逐周期重复。
    streams = [stream(0, 30), stream(30, 0)]
    assert compute_conflict_audit(streams, [(0, 1, 5)], 0, 120) == [
        [[0, 5], [30, 35], [60, 65], [90, 95]]
    ]


def test_clearance_boundary_exactly_d_seconds_is_safe():
    # s0 [0,20)，s1 [25,45)：间隔 5 秒。d=4 时 5 > 4 安全；d=5 时
    # s1 在 t=25 距 s0 结束恰 5 秒，"不足 5 秒"不成立，仍安全。
    streams = [stream(0, 20), stream(25, 45)]
    assert compute_conflict_audit(streams, [(0, 1, 4)], 0, 60) == [[]]
    assert compute_conflict_audit(streams, [(0, 1, 5)], 0, 60) == [[]]
    # d=6 时 [25,45) 内 t-20 < 6 ⇔ t < 26，仅 [25,26) 危险
    assert compute_conflict_audit(streams, [(0, 1, 6)], 0, 60) == [[[25, 26]]]


def test_both_tails_overlap_is_not_dangerous():
    # s0 [0,10)，s1 [20,30)，d=30：s1 全绿段 s0 结束 10..25 秒前均不足 30，
    # 故 [20,30) 危险；但 t∈[30,40) 两股都红（车尾段相互重叠），不危险。
    streams = [stream(0, 10), stream(20, 30)]
    assert compute_conflict_audit(streams, [(0, 1, 30)], 0, 60) == [
        [[20, 30]]
    ]


def test_green_before_range_start_contributes_its_tail():
    # s0 绿灯在 t=100（范围起点，不含）结束，s1 [105,120)；d=10 时
    # t∈[105,110) s1 绿且 s0 结束不足 10 秒；d=5 时恰 5 秒起安全。
    streams = [
        stream(cycle=200, green_start=0, green_end=100),
        stream(cycle=200, green_start=105, green_end=120),
    ]
    assert compute_conflict_audit(streams, [(0, 1, 10)], 100, 200) == [
        [[105, 110]]
    ]
    assert compute_conflict_audit(streams, [(0, 1, 5)], 100, 200) == [[]]


def test_range_start_inside_a_green_is_clipped():
    # s0 [0,30) 周期 60，s1 常绿；范围 [10,70)：s0 绿 [10,30)、[60,70)
    streams = [stream(0, 30), stream(0, 0)]
    assert compute_conflict_audit(streams, [(0, 1, 0)], 10, 70) == [
        [[10, 30], [60, 70]]
    ]
    # d=100 时 s0 的清空尾部覆盖红灯间隙：s1 全程绿，s0 绿灯结束后 100 秒内
    # s1 仍绿即危险（[30,130) 全在尾部内），整个范围 [10,70) 均危险。
    assert compute_conflict_audit(streams, [(0, 1, 100)], 10, 70) == [
        [[10, 70]]
    ]


def test_wrapping_green_intervals_merge_across_zero():
    # s0 跨零点 [50,60)∪[0,10)，s1 [8,20)，d=0：同时绿恰为 [8,10)
    streams = [stream(50, 10), stream(8, 20)]
    assert compute_conflict_audit(streams, [(0, 1, 0)], 0, 60) == [[[8, 10]]]
    # d=3：s1 在 [8,13) 时 s0 的 [0,10) 段结束不足 3 秒（[10,13)）
    assert compute_conflict_audit(streams, [(0, 1, 3)], 0, 60) == [
        [[8, 13]]
    ]


def test_offset_shifts_green_phases():
    # s0 offset=10 绿本地 [0,30)：绝对 [50,60)∪[0,20)；s1 [0,30)
    streams = [stream(offset=10), stream()]
    assert compute_conflict_audit(streams, [(0, 1, 0)], 0, 120) == [
        [[0, 20], [60, 80]]
    ]


def test_different_cycles():
    streams = [stream(cycle=60, green_end=30), stream(cycle=90, green_end=45)]
    # 同时绿：t 同时落在 [0,30) mod 60 与 [0,45) mod 90
    assert compute_conflict_audit(streams, [(0, 1, 0)], 0, 360) == [
        [[0, 30], [120, 135], [180, 210], [300, 315]]
    ]


def test_always_green_pair_is_dangerous_throughout_range():
    streams = [
        stream(cycle=60, green_start=0, green_end=0),
        stream(cycle=90, green_start=0, green_end=0),
    ]
    assert compute_conflict_audit(streams, [(0, 1, 10)], 100, 200) == [
        [[100, 200]]
    ]


def test_safe_pair_returns_empty_array():
    # 两股各占周期一半且 d=0：[0,30) 与 [30,60)，端点相接不重叠
    streams = [stream(0, 30), stream(30, 0)]
    assert compute_conflict_audit(streams, [(0, 1, 0)], 0, 120) == [[]]


def test_multiple_pairs_returned_in_input_order_and_normalized():
    streams = [
        stream(0, 30),
        stream(10, 40),
        stream(40, 50),
    ]
    pairs = [(2, 0, 0), (0, 1, 0)]
    results = compute_conflict_audit(streams, pairs, 0, 60)
    assert len(results) == 2
    # 顺序与请求一致（先 (2,0) 后 (0,1)），下标规范化由 API 层负责
    assert results[0] == []  # [40,50) 与 [0,30) 不相交
    assert results[1] == [[10, 30]]


def test_mixed_clearances_share_global_enumeration():
    # 同一批请求中各对 clearance 不同，结果须按各自 d 计算
    streams = [
        stream(0, 20),
        stream(25, 45),
        stream(50, 55),
    ]
    results = compute_conflict_audit(
        streams, [(0, 1, 6), (1, 2, 100)], 0, 60
    )
    assert results[0] == [[25, 26]]  # 间隔 5、d=6
    # s1 [25,45) 与 s2 [50,55)：d=100 时双向尾部都很长——s2 绿时 s1 结束
    # 5..10 秒前 ([50,55))；s1 绿时 s2 上周期绿灯 [-10,-5) 也结束不足 100 秒
    # （[25,45) 时距其结束 30..50 秒），故两段都危险。
    assert results[1] == [[25, 45], [50, 55]]


# ---------- 随机参数对照 ----------


def random_stream(rng):
    cycle = rng.randint(1, 120)
    offset = rng.randint(0, cycle - 1)
    green_start = rng.randint(0, cycle - 1)
    green_end = rng.randint(0, cycle - 1)
    return stream(
        cycle=cycle, offset=offset, green_start=green_start, green_end=green_end
    )


def test_random_cases_match_second_by_second_reference():
    rng = random.Random(20260925)
    for _ in range(1500):
        count = rng.randint(2, 5)
        streams = [random_stream(rng) for _ in range(count)]
        audit_from = rng.randint(0, 300)
        audit_to = audit_from + rng.randint(1, 400)
        pair_count = rng.randint(1, min(4, count * (count - 1) // 2))
        keys = set()
        pairs = []
        while len(pairs) < pair_count:
            i = rng.randrange(count)
            j = rng.randrange(count)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in keys:
                continue
            keys.add(key)
            pairs.append((i, j, rng.randint(0, 20)))
        expected = brute_force(streams, pairs, audit_from, audit_to)
        actual = compute_conflict_audit(streams, pairs, audit_from, audit_to)
        assert actual == expected


def test_random_large_clearance_matches_reference():
    rng = random.Random(4242)
    for _ in range(200):
        count = rng.randint(2, 4)
        streams = [random_stream(rng) for _ in range(count)]
        audit_from = rng.randint(0, 100)
        audit_to = audit_from + rng.randint(1, 300)
        pairs = [
            (0, 1, rng.choice([0, 1, 50, 200, 300]))
        ]
        expected = brute_force(streams, pairs, audit_from, audit_to)
        actual = compute_conflict_audit(streams, pairs, audit_from, audit_to)
        assert actual == expected


def test_full_day_range_performance():
    # 全天 86400 秒、小周期：区间枚举规模有界，毫秒级完成
    import time

    streams = [
        stream(cycle=60, offset=0, green_start=0, green_end=30),
        stream(cycle=90, offset=7, green_start=10, green_end=50),
    ]
    started = time.perf_counter()
    results = compute_conflict_audit(streams, [(0, 1, 300)], 0, 86400)
    assert time.perf_counter() - started < 2.0
    assert results[0]
    assert results[0][0][0] >= 0
    assert results[0][-1][1] <= 86400
