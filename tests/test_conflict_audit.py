"""冲突车流安全间隔审计核心算法测试。

确定性用例覆盖：两股同时绿灯、清空间隔尾部单向延长、间隔恰为 c 秒安全、
c=0、范围起点之前绿灯尾部拖入、跨零点绿灯、常绿、不同周期、半开范围边界
与相接区间合并。另含 2000 组随机参数下与逐整数秒（含向范围前回溯判"绿灯
结束尚不足 c 秒"）暴力参考实现的逐项对照。
"""

import random

from app.greenwave import compute_conflict_audit


def is_green_at(stream, moment):
    """绝对整数时刻 moment 在该股车流是否为绿灯（与 is_green 同语义）。"""
    cycle, offset = stream[0], stream[1]
    green_start, green_end = stream[2], stream[3]
    if green_start == green_end:
        return True
    phase = (moment + offset) % cycle
    if green_start < green_end:
        return green_start <= phase < green_end
    return phase >= green_start or phase < green_end


def brute_force(streams, audit_from, audit_to, pairs):
    """逐整数秒判定的参考实现，返回与 pairs 同序的极大半开危险区间列表。

    先把每股在扩大范围 [audit_from - max_c, audit_to) 内的灯色预展开为字节
    数组（下标偏移），"绿灯结束尚不足 c 秒"只需 O(1) 查找前 c 秒灯色。
    """
    max_clearance = max((pair[2] for pair in pairs), default=0)
    lower = audit_from - max_clearance
    # 每股灯色的前缀和：prefix[k+1] - prefix[k] 为 marks[k]，区间绿秒数 O(1)
    # 求得，"绿灯结束尚不足 c 秒"即当秒红而 [now-c, now) 内存在绿灯秒。
    prefixes = []
    for stream in streams:
        marks = bytearray(audit_to - lower)
        for moment in range(lower, audit_to):
            marks[moment - lower] = is_green_at(stream, moment)
        prefix = [0] * (len(marks) + 1)
        for i, value in enumerate(marks):
            prefix[i + 1] = prefix[i] + value
        prefixes.append(prefix)

    def green(index, moment):
        pos = moment - lower
        return prefixes[index][pos + 1] - prefixes[index][pos] == 1

    def recently_ended(index, moment, clearance):
        if clearance <= 0 or green(index, moment):
            return False
        prefix = prefixes[index]
        end = moment - lower
        start = end - clearance
        return prefix[end] - prefix[start] > 0

    results = []
    for index_a, index_b, clearance in pairs:
        intervals: list[list[int]] = []
        for now in range(audit_from, audit_to):
            ga, gb = green(index_a, now), green(index_b, now)
            dangerous = (
                (ga and gb)
                or (ga and recently_ended(index_b, now, clearance))
                or (gb and recently_ended(index_a, now, clearance))
            )
            if dangerous:
                if intervals and intervals[-1][1] == now:
                    intervals[-1][1] = now + 1
                else:
                    intervals.append([now, now + 1])
        results.append(intervals)
    return results


def random_stream(rng):
    cycle = rng.randint(1, 40)
    green_start = rng.randrange(cycle)
    # 约 1/8 概率常绿，否则随机另一个绿灯端点（可能跨零点）
    if cycle > 1 and rng.random() < 0.125:
        green_end = green_start
    else:
        green_end = rng.randrange(cycle)
    return (cycle, rng.randrange(cycle), green_start, green_end)


def solve(streams, audit_from, audit_to, pairs):
    return compute_conflict_audit(
        [
            {
                "cycle": cycle,
                "offset": offset,
                "green_start": green_start,
                "green_end": green_end,
            }
            for cycle, offset, green_start, green_end in streams
        ],
        audit_from,
        audit_to,
        pairs,
    )


# ---------- 确定性用例 ----------


def test_overlapping_greens_plus_tails():
    # A=[0,30)、B=[20,50)、c=5：重叠 [20,30) 与 B 绿灯撞 A 尾部的 [30,35)
    streams = [(60, 0, 0, 30), (60, 0, 20, 50)]
    assert solve(streams, 0, 120, [(0, 1, 5)]) == [
        [[20, 35], [80, 95]]
    ]


def test_clearance_zero_is_simultaneous_green_only():
    # 首尾相接的两段绿灯：c=0 时无任何同时绿灯，安全
    streams = [(60, 0, 0, 30), (60, 0, 30, 50)]
    assert solve(streams, 0, 120, [(0, 1, 0)]) == [[]]


def test_gap_less_than_clearance_is_dangerous():
    # A=[0,30)、B=[30,50)、c=10：B 绿灯的前 10 秒撞上 A 的尾部
    streams = [(60, 0, 0, 30), (60, 0, 30, 50)]
    assert solve(streams, 0, 120, [(0, 1, 10)]) == [
        [[30, 40], [90, 100]]
    ]


def test_gap_equal_to_clearance_is_safe():
    # A=[0,20)、B=[30,50)：A 尾延到 30，B 30 启绿，半开端点恰好不撞
    streams = [(60, 0, 0, 20), (60, 0, 30, 50)]
    assert solve(streams, 0, 120, [(0, 1, 10)]) == [[]]


def test_green_before_range_contributes_its_tail():
    # 范围从 32 开始：[0,30) 的尾部延到 40，裁剪为 [32,40)
    streams = [(60, 0, 0, 30), (60, 0, 30, 50)]
    assert solve(streams, 32, 120, [(0, 1, 10)]) == [
        [[32, 40], [90, 100]]
    ]


def test_tail_only_at_range_start():
    # A=[0,30)、B=[35,55)、c=10：范围 [35,...) 首段 [35,40) 纯由
    # 起点之前结束的 A 绿灯尾部造成
    streams = [(60, 0, 0, 30), (60, 0, 35, 55)]
    assert solve(streams, 35, 120, [(0, 1, 10)]) == [
        [[35, 40], [60, 65], [95, 100]]
    ]


def test_wrapping_green_with_equal_clearance_safe():
    # A 跨零点 [50,60)∪[0,10)、B=[20,40)：两侧间隔都恰为 c=5 秒，安全
    streams = [(60, 0, 50, 10), (60, 0, 20, 40)]
    assert solve(streams, 0, 120, [(0, 1, 5)]) == [[]]


def test_wrapping_green_tail_overlap():
    # 同上但 c=6：A 尾延到 16 撞 B 绿灯 [20,..)? 16<20 不撞；
    # B 尾延到 46，A 50 启绿，46<50 也不撞——扩大到 c=11：
    # A 尾延到 21 撞 B 绿灯 [20,21)；B 尾延到 51 撞 A 绿灯 [50,51)
    streams = [(60, 0, 50, 10), (60, 0, 20, 40)]
    assert solve(streams, 0, 120, [(0, 1, 11)]) == [
        [[20, 21], [50, 51], [80, 81], [110, 111]]
    ]


def test_always_green_stream_never_has_tail():
    # A 常绿：危险仅为 B 绿灯与其尾部，A 自身没有"绿灯结束"
    streams = [(60, 0, 0, 0), (60, 0, 20, 40)]
    assert solve(streams, 0, 120, [(0, 1, 10)]) == [
        [[20, 50], [80, 110]]
    ]


def test_both_streams_always_green():
    streams = [(60, 0, 0, 0), (60, 0, 0, 0)]
    assert solve(streams, 10, 70, [(0, 1, 30)]) == [[[10, 70]]]


def test_different_cycles():
    # A 周期 60 绿灯 [0,20)、B 周期 40 绿灯 [10,30)、c=5
    streams = [(60, 0, 0, 20), (40, 0, 10, 30)]
    got = solve(streams, 0, 120, [(0, 1, 5)])
    assert got == brute_force(streams, 0, 120, [(0, 1, 5)])


def test_range_end_is_exclusive():
    # 危险恰延伸至范围终点：最后一个半开区间右端等于 audit_to
    streams = [(60, 0, 0, 30), (60, 0, 20, 50)]
    assert solve(streams, 0, 35, [(0, 1, 5)]) == [[[20, 35]]]
    # 终点再缩一秒则右端随之裁剪
    assert solve(streams, 0, 34, [(0, 1, 5)]) == [[[20, 34]]]


def test_offset_shifts_phases():
    # A offset=10：绝对绿灯为 [-10,20) mod 60 = [50,60)∪[0,20)；
    # B=[20,50)、c=10：A 尾延到 30 撞 B 绿灯 [20,30)；A 的 [50,60) 绿灯
    # 又撞上 B 延到 60 的尾部 [50,60)
    streams = [(60, 10, 0, 30), (60, 0, 20, 50)]
    assert solve(streams, 0, 120, [(0, 1, 10)]) == [
        [[20, 30], [50, 60], [80, 90], [110, 120]]
    ]


def test_multiple_pairs_returned_in_input_order():
    streams = [
        (60, 0, 0, 30),
        (60, 0, 20, 50),
        (60, 0, 40, 55),
    ]
    pairs = [(0, 2, 0), (0, 1, 5), (1, 2, 10)]
    got = solve(streams, 0, 60, pairs)
    assert got == brute_force(streams, 0, 60, pairs)
    assert len(got) == 3
    # 第一对 c=0：A=[0,30) 与 C=[40,55) 从不重叠
    assert got[0] == []


def test_empty_pairs_returns_empty_results():
    streams = [(60, 0, 0, 30), (60, 0, 20, 50)]
    assert solve(streams, 0, 60, []) == []


# ---------- 随机参数对照 ----------


def test_random_against_brute_force():
    rng = random.Random(20260925)
    for _ in range(2000):
        stream_count = rng.randint(2, 8)
        streams = [random_stream(rng) for _ in range(stream_count)]
        audit_from = rng.randint(0, 200)
        audit_to = audit_from + rng.randint(1, 400)
        all_pairs = [
            (i, j)
            for i in range(stream_count)
            for j in range(i + 1, stream_count)
        ]
        rng.shuffle(all_pairs)
        chosen = all_pairs[: rng.randint(0, min(4, len(all_pairs)))]
        pairs = [
            (i, j, rng.randint(0, 5)) for i, j in chosen
        ]
        got = solve(streams, audit_from, audit_to, pairs)
        expected = brute_force(streams, audit_from, audit_to, pairs)
        assert got == expected, (
            f"mismatch streams={streams} range=({audit_from},{audit_to}) "
            f"pairs={pairs} got={got} expected={expected}"
        )


def test_random_full_day_range_with_large_clearance():
    # 全天范围 + 大周期 + 300 秒清空间隔的压力用例
    rng = random.Random(4242)
    for _ in range(12):
        stream_count = rng.randint(2, 8)
        streams = [
            (
                rng.choice((60, 90, 120, 3600, 86400)),
                0,  # offset 在下方单独随机，避免超过周期
                0,
                0,
            )
            for _ in range(stream_count)
        ]
        built = []
        for cycle, _o, _gs, _ge in streams:
            gs = rng.randrange(cycle)
            ge = gs if rng.random() < 0.2 else rng.randrange(cycle)
            built.append((cycle, rng.randrange(cycle), gs, ge))
        streams = built
        pairs = [
            (i, j, rng.choice((0, 1, 60, 300)))
            for i in range(stream_count)
            for j in range(i + 1, stream_count)
        ]
        got = solve(streams, 0, 86400, pairs)
        expected = brute_force(streams, 0, 86400, pairs)
        assert got == expected
