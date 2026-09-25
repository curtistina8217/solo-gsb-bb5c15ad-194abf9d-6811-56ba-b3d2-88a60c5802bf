"""路口放行能力测算核心算法测试。

确定性用例覆盖：常绿与普通绿灯下的最小放行量、红灯不放行、deadline 含端点
（恰等于 deadline 通过可行）、deadline 早于下一次清空、同秒多批按输入顺序
入队、期初队列、跨零点绿灯、半开范围终点（最后一秒可通过）、无车辆、
deadline 早于到达秒等无解情形，以及大车辆数不逐辆展开。
另含 3000 组随机参数下与逐辆展开暴力参考实现（逐放行量扫描）的逐项对照。
"""

import random
from collections import deque

from app.greenwave import compute_min_discharge


def brute_force(
    cycle,
    offset,
    green_start,
    green_end,
    forecast_from,
    forecast_to,
    initial_queue,
    arrivals,
):
    """逐辆展开 + 逐放行量扫描的参考实现，返回 (最小放行量, 通过秒) 或 None。"""

    def green(now):
        if green_start == green_end:
            return True
        phase = (now + offset) % cycle
        if green_start < green_end:
            return green_start <= phase < green_end
        return phase >= green_start or phase < green_end

    total = initial_queue + sum(count for _t, count, _d in arrivals)

    def simulate(discharge):
        queue = deque([-1] * initial_queue)
        by_second = {}
        for index, (moment, _count, _deadline) in enumerate(arrivals):
            by_second.setdefault(moment, []).append(index)
        remaining = [count for _moment, count, _d in arrivals]
        pass_times = [None] * len(arrivals)
        for now in range(forecast_from, forecast_to):
            for index in by_second.get(now, []):
                queue.extend([index] * arrivals[index][1])
            if queue and green(now):
                for _ in range(min(discharge, len(queue))):
                    batch = queue.popleft()
                    if batch >= 0:
                        remaining[batch] -= 1
                        if remaining[batch] == 0:
                            pass_times[batch] = now
            for index, (_moment, _count, deadline) in enumerate(arrivals):
                if deadline == now and pass_times[index] is None:
                    return None
        if queue:
            return None
        if any(value is None for value in pass_times):
            return None
        return list(pass_times)

    if total == 0:
        return 1, []
    for discharge in range(1, total + 1):
        result = simulate(discharge)
        if result is not None:
            return discharge, result
    return None


def solve(**kw):
    """以常绿默认值调用 compute_min_discharge，关键字覆盖。"""
    params = {
        "cycle": 60,
        "offset": 0,
        "green_start": 0,
        "green_end": 0,  # 常绿
        "forecast_from": 0,
        "forecast_to": 30,
        "initial_queue": 0,
        "arrivals": [],
    }
    params.update(kw)
    return compute_min_discharge(**params)


def batch(moment, count, deadline):
    return (moment, count, deadline)


# ---------- 确定性用例 ----------


def test_always_green_single_batch_needs_one_per_second():
    # 常绿：t=0 到 3 辆，范围 [0,5)；d=1 通过秒 0,1,2，最小放行量为 1
    assert solve(arrivals=[batch(0, 3, 4)]) == (1, [2])


def test_discharge_greater_than_one_when_backlogged():
    # 常绿：期初 4 辆，范围 [0,3) 只有 3 个绿灯秒，d=2 恰在 t=1 清空
    assert solve(forecast_to=3, initial_queue=4, arrivals=[]) == (2, [])


def test_initial_queue_blocks_batch_until_drained():
    # 常绿：期初 3 辆，t=2 到 1 辆、deadline=2；d=1 时期初 t=0..2 才走完，
    # 批次 t=3 通过违约；d=2 时期初 t=0..1 清空、批次 t=2 当秒通过 ⇒ 最小 2
    result = solve(forecast_to=8, initial_queue=3, arrivals=[batch(2, 1, 2)])
    assert result == (2, [2])


def test_deadline_is_inclusive():
    # 常绿：t=0 到 2 辆，deadline=1：d=1 时最后一辆 t=1 通过（恰为 deadline）可行
    assert solve(forecast_to=5, arrivals=[batch(0, 2, 1)]) == (1, [1])
    # deadline=0 则两批无法在同一秒以 d=1 走完，需要 d=2
    assert solve(forecast_to=5, arrivals=[batch(0, 2, 0)]) == (2, [0])


def test_range_end_is_exclusive():
    # 常绿：t=0 到 3 辆，范围 [0,3)，d=1 通过秒 0,1,2 恰清空
    assert solve(forecast_to=3, arrivals=[batch(0, 3, 2)]) == (1, [2])
    # 4 辆则 d=1 最后一辆要到 t=3（范围外），d=2 才在 t=1 清空
    assert solve(forecast_to=3, arrivals=[batch(0, 4, 2)]) == (2, [1])


def test_red_light_never_discharges():
    # 绿灯 [0,10)：t=20 到达的批次在范围 [0,60) 内一直遇红，无解
    assert solve(
        green_start=0,
        green_end=10,
        forecast_to=60,
        arrivals=[batch(20, 3, 59)],
    ) is None


def test_red_gap_sets_minimum_discharge():
    # 绿灯 [0,30)，范围 [0,60)：t=0 到 10 辆、deadline=4（红灯前通过），
    # 需要 ceil(10/5) = 2
    result = solve(
        green_start=0,
        green_end=30,
        forecast_to=60,
        arrivals=[batch(0, 10, 4)],
    )
    assert result == (2, [4])
    # deadline=9 时 d=1 也能在 t=9 走完
    result = solve(
        green_start=0,
        green_end=30,
        forecast_to=60,
        arrivals=[batch(0, 10, 9)],
    )
    assert result == (1, [9])


def test_missed_green_waits_next_cycle():
    # 绿灯 [0,10)：t=5 到 8 辆、deadline=65，第一绿段只能走 5 辆（t=5..9），
    # 剩余 3 辆下一周期 t=60..62 通过；d=1 即可
    result = solve(
        green_start=0,
        green_end=10,
        forecast_to=70,
        arrivals=[batch(5, 8, 65)],
    )
    assert result == (1, [62])
    # deadline=60（含端点）：t=60 当秒通过的那辆不算最后一辆，需要 d=2，
    # 第一绿段 t=5..9 走 10 容量，8 辆在 t=8 走完
    result = solve(
        green_start=0,
        green_end=10,
        forecast_to=70,
        arrivals=[batch(5, 8, 60)],
    )
    assert result == (2, [8])


def test_same_second_batches_input_order_and_shared_capacity():
    # 常绿、d=1：t=0 两批（2 辆、1 辆，deadline 均宽松）按输入顺序通过秒为 1、2
    assert solve(
        forecast_to=10,
        arrivals=[batch(0, 2, 9), batch(0, 1, 9)],
    ) == (1, [1, 2])
    # 第二批 deadline=1：d=1 时它 t=2 通过违约；d=2 时第一批 t=0 走完、
    # 第二批 t=1 通过，恰满足含端点 deadline
    assert solve(
        forecast_to=10,
        arrivals=[batch(0, 2, 9), batch(0, 1, 1)],
    ) == (2, [0, 1])
    # 第二批 deadline=0：d=2 时当秒容量被第一批占满，第二批 t=1 才通过；
    # d=3 时同秒 3 辆全部通过
    assert solve(
        forecast_to=10,
        arrivals=[batch(0, 2, 9), batch(0, 1, 0)],
    ) == (3, [0, 0])


def test_wrapping_green_discharge():
    # 跨零点绿灯 [50,60)∪[0,10)，范围 [50,70)：t=55 到 4 辆、deadline=56，
    # d=1 时 t=55..58 走完、最后一辆 t=58 违约；d=2 时 t=55、56 走完
    result = solve(
        green_start=50,
        green_end=10,
        forecast_from=50,
        forecast_to=70,
        arrivals=[batch(55, 4, 56)],
    )
    assert result == (2, [56])
    # deadline=55（含端点）：当秒只放 2 辆，需 d=4
    result = solve(
        green_start=50,
        green_end=10,
        forecast_from=50,
        forecast_to=70,
        arrivals=[batch(55, 4, 55)],
    )
    assert result == (4, [55])


def test_offset_shifts_green_window():
    # offset=30、绿灯 [0,10)：绝对绿段为 t%60 ∈ [30,40)
    result = solve(
        offset=30,
        green_start=0,
        green_end=10,
        forecast_from=30,
        forecast_to=45,
        arrivals=[batch(35, 4, 44)],
    )
    # 绿秒 35..39 共 5 秒，d=1 最后一辆 t=38 通过
    assert result == (1, [38])


def test_no_vehicles_returns_one():
    assert solve() == (1, [])
    assert solve(initial_queue=0, arrivals=[]) == (1, [])


def test_arrival_at_last_second_with_tight_deadline():
    # 范围最后一秒 t=9 到达 3 辆、deadline=9：常绿时需 d=3
    assert solve(forecast_to=10, arrivals=[batch(9, 3, 9)]) == (3, [9])
    # 最后一秒为红灯则无解
    assert solve(
        green_start=0,
        green_end=9,
        forecast_to=10,
        arrivals=[batch(9, 1, 9)],
    ) is None


def test_large_counts_without_per_vehicle_expansion():
    # 10^12 辆在常绿范围 [0,1000) 内：最小放行量 ceil(10^12 / 1000) = 10^9
    result = solve(
        forecast_to=1000,
        arrivals=[batch(0, 10**12, 999)],
    )
    assert result == (10**9, [999])
    # 期初 10^6 辆同批竞争：需要 ceil((10^12 + 10^6) / 1000)
    result = solve(
        forecast_to=1000,
        initial_queue=10**6,
        arrivals=[batch(0, 10**12, 999)],
    )
    needed = (10**12 + 10**6 + 999) // 1000
    assert result[0] == needed
    assert result[1] == [999]


def test_multi_batch_deadlines_increase_required_discharge():
    # 常绿范围 [0,20)：t=0 到 6 辆、t=2 到 6 辆。两批 deadline 均宽松时
    # d=1 即可（第一批 t=5、第二批 t=11 通过）。
    result = solve(
        forecast_to=20,
        arrivals=[batch(0, 6, 10), batch(2, 6, 15)],
    )
    assert result == (1, [5, 11])
    # 第二批 deadline=3：d=1 时第二批 t=8..11 通过违约；d=2 时第一批
    # t=0..2 走完、第二批 t=3..5 通过，最后一辆 t=5 仍晚于 deadline=3；
    # d=3 时第一批 t=0、1 走完，第二批 t=2、3 走完（最后一辆恰为 deadline）。
    result = solve(
        forecast_to=20,
        arrivals=[batch(0, 6, 10), batch(2, 6, 3)],
    )
    assert result == (3, [1, 3])
    # 两批 deadline 都很紧（第一批 deadline=0、第二批 deadline=3）：d=3 时
    # 第一批 t=0 只能走 3 辆、最后一辆 t=1 违约，需要 d=6：第一批 t=0 走完、
    # 第二批 t=2 走 6 辆容量中的 6 辆——t=2 当秒走完，[0, 2]。
    result = solve(
        forecast_to=20,
        arrivals=[batch(0, 6, 0), batch(2, 6, 3)],
    )
    assert result == (6, [0, 2])


# ---------- 随机对照：逐辆展开 + 逐放行量暴力实现 ----------


def test_random_against_per_vehicle_brute_force():
    rng = random.Random(20260925)
    for _ in range(3000):
        cycle = rng.choice([1, 2, 3, 7, 12, 60])
        offset = rng.randrange(cycle)
        kind = rng.randrange(3)
        if kind == 0:  # 常绿
            green_start = green_end = rng.randrange(cycle)
        elif kind == 1:  # 普通绿灯
            green_start = rng.randrange(cycle)
            green_end = rng.randrange(cycle)
            if green_start == green_end:
                green_end = (green_end + 1) % cycle
            if green_start > green_end:
                green_start, green_end = green_end, green_start
        else:  # 跨零点绿灯
            green_start = rng.randrange(cycle)
            green_end = rng.randrange(cycle)
            if green_start == green_end:
                green_start = (green_start + 1) % cycle
            if green_start < green_end:
                green_start, green_end = green_end, green_start
        forecast_from = rng.randrange(0, 20)
        forecast_to = forecast_from + rng.randrange(1, 40)
        initial = rng.randrange(0, 5)
        arrivals = []
        for _ in range(rng.randrange(0, 6)):
            moment = rng.randrange(forecast_from, forecast_to)
            arrivals.append(
                (moment, rng.randrange(1, 5), rng.randrange(moment, forecast_to))
            )
        expected = brute_force(
            cycle, offset, green_start, green_end,
            forecast_from, forecast_to, initial, arrivals,
        )
        actual = compute_min_discharge(
            cycle=cycle,
            offset=offset,
            green_start=green_start,
            green_end=green_end,
            forecast_from=forecast_from,
            forecast_to=forecast_to,
            initial_queue=initial,
            arrivals=arrivals,
        )
        assert actual == expected
