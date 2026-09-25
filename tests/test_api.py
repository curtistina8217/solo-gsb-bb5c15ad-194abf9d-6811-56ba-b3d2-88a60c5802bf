import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.main import (
    green_wave_conflict_audit,
    green_wave_convoy_starts,
    green_wave_earliest_route,
    green_wave_max_coverage,
    green_wave_min_discharge,
    green_wave_offset_diagnostics,
    green_wave_queue_forecast,
    green_wave_slot_offsets,
    green_wave_timetable_reschedule,
    green_wave_windows,
)
from app.schemas import (
    ConflictAuditRequest,
    ConvoyStartsRequest,
    EarliestRouteRequest,
    GreenWaveRequest,
    Intersection,
    MinDischargeRequest,
    OffsetDiagnosticsRequest,
    QueueForecastRequest,
    SlotOffsetsRequest,
    TimetableRescheduleRequest,
)


def make_request(payload):
    return GreenWaveRequest.model_validate(payload)


def test_api_returns_windows_payload():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 300,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30, "travel": 0},
                {"cycle": 60, "offset": 30, "green_start": 0, "green_end": 30, "travel": 30},
            ],
        }
    )
    windows = green_wave_windows(req)
    assert [[w.start, w.end] for w in windows] == [
        [0, 30], [60, 90], [120, 150], [180, 210], [240, 270],
    ]


def test_api_empty_array_on_no_solution():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 10, "travel": 0},
                {"cycle": 60, "offset": 0, "green_start": 20, "green_end": 30, "travel": 0},
            ],
        }
    )
    assert green_wave_windows(req) == []


def test_reject_invalid_cycle():
    with pytest.raises(ValidationError):
        Intersection(cycle=0, offset=0, green_start=0, green_end=10, travel=0)


def test_reject_negative_travel():
    with pytest.raises(ValidationError):
        Intersection(cycle=60, offset=0, green_start=0, green_end=10, travel=-1)


def test_reject_offset_outside_cycle():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "intersections": [
                {"cycle": 60, "offset": 60, "green_start": 0, "green_end": 10, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        req.validate_ranges()
    assert exc.value.status_code == 422


def test_reject_green_endpoint_outside_cycle():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 60, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        req.validate_ranges()
    assert exc.value.status_code == 422


def test_reject_range_over_one_hour():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 3601,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 10, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        req.validate_ranges()
    assert exc.value.status_code == 422


def test_reject_too_many_intersections():
    with pytest.raises(ValidationError):
        make_request(
            {
                "depart_from": 0,
                "depart_to": 60,
                "intersections": [
                    {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 10}
                    for _ in range(9)
                ],
            }
        )


def test_reject_empty_intersections():
    with pytest.raises(ValidationError):
        make_request({"depart_from": 0, "depart_to": 60, "intersections": []})


# ---------- travel_max 波动行驶时间 ----------

def test_travel_max_shrinks_windows():
    # 绿灯 [20,40)，行驶时间 [0,10]：窗口由 20 秒收缩为 10 秒
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 120,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 20,
                 "green_end": 40, "travel": 0, "travel_max": 10}
            ],
        }
    )
    windows = green_wave_windows(req)
    assert [[w.start, w.end] for w in windows] == [[20, 30], [80, 90]]


def test_travel_max_defaults_to_travel():
    base = {
        "cycle": 60, "offset": 0, "green_start": 20, "green_end": 40,
        "travel": 5,
    }
    req = make_request(
        {"depart_from": 0, "depart_to": 120, "intersections": [base]}
    )
    req2 = make_request(
        {"depart_from": 0, "depart_to": 120,
         "intersections": [{**base, "travel_max": 5}]}
    )
    assert green_wave_windows(req) == green_wave_windows(req2)


def test_reject_travel_max_less_than_travel():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 30, "travel": 20, "travel_max": 19}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        req.validate_ranges()
    assert exc.value.status_code == 422


def test_reject_travel_max_over_86400():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 30, "travel": 0, "travel_max": 86401}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        req.validate_ranges()
    assert exc.value.status_code == 422


def test_travel_max_86400_boundary_accepted():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "intersections": [
                # 整周期常绿：波动再大也不受影响
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 0, "travel": 0, "travel_max": 86400}
            ],
        }
    )
    req.validate_ranges()
    windows = green_wave_windows(req)
    assert [[w.start, w.end] for w in windows] == [[0, 60]]


# ---------- POST /api/green-wave/max-coverage ----------

def test_max_coverage_payload_overlap():
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 120,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 40, "travel": 0},
                {"cycle": 60, "offset": 0, "green_start": 20, "green_end": 50, "travel": 0},
            ],
        }
    )
    resp = green_wave_max_coverage(req)
    assert resp.max_coverage == 2
    assert [[w.start, w.end] for w in resp.optimal_intervals] == [
        [20, 40], [80, 100],
    ]


def test_max_coverage_zero_returns_complete_range():
    req = make_request(
        {
            "depart_from": 10,
            "depart_to": 70,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 20, "green_end": 40,
                 "travel": 0, "travel_max": 30}
            ],
        }
    )
    resp = green_wave_max_coverage(req)
    assert resp.max_coverage == 0
    assert [[w.start, w.end] for w in resp.optimal_intervals] == [[10, 70]]


def test_max_coverage_reuses_request_validation():
    # 与 /windows 相同的跨字段校验：发车范围超过 3600 秒仍返回 422。
    req = make_request(
        {
            "depart_from": 0,
            "depart_to": 3601,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 10, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_max_coverage(req)
    assert exc.value.status_code == 422


def test_max_coverage_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/green-wave/max-coverage" in paths
    assert "post" in paths["/api/green-wave/max-coverage"]


# ---------- POST /api/green-wave/offset-diagnostics ----------

def make_diag_request(payload):
    return OffsetDiagnosticsRequest.model_validate(payload)


def test_offset_diagnostics_payload_plain_green():
    req = make_diag_request(
        {
            "cycle": 60,
            "green_start": 20,
            "green_end": 40,
            "observations": [
                {"arrival_time": 0, "observed_green": True},
                {"arrival_time": 5, "observed_green": True},
            ],
        }
    )
    resp = green_wave_offset_diagnostics(req)
    # arrival=0 观测绿 ⇔ offset∈[20,40)；arrival=5 观测绿 ⇔ offset∈[15,35)
    # 零错配集合为两者之交 [20,35)
    assert resp.min_mismatches == 0
    assert [[w.start, w.end] for w in resp.optimal_intervals] == [[20, 35]]


def test_offset_diagnostics_payload_wrapping_optimal_set():
    req = make_diag_request(
        {
            "cycle": 60,
            "green_start": 50,
            "green_end": 10,
            "observations": [{"arrival_time": 0, "observed_green": True}],
        }
    )
    resp = green_wave_offset_diagnostics(req)
    assert resp.min_mismatches == 0
    assert [[w.start, w.end] for w in resp.optimal_intervals] == [
        [0, 10], [50, 60]
    ]


def test_offset_diagnostics_min_mismatches_positive():
    # 常绿路口但观测到红灯：任何 offset 都错 1 条
    req = make_diag_request(
        {
            "cycle": 60,
            "green_start": 30,
            "green_end": 30,
            "observations": [{"arrival_time": 9, "observed_green": False}],
        }
    )
    resp = green_wave_offset_diagnostics(req)
    assert resp.min_mismatches == 1
    assert [[w.start, w.end] for w in resp.optimal_intervals] == [[0, 60]]


def test_offset_diagnostics_accepts_negative_arrival_time():
    req = make_diag_request(
        {
            "cycle": 60,
            "green_start": 20,
            "green_end": 40,
            "observations": [{"arrival_time": -1, "observed_green": True}],
        }
    )
    resp = green_wave_offset_diagnostics(req)
    assert resp.min_mismatches == 0
    assert [[w.start, w.end] for w in resp.optimal_intervals] == [[21, 41]]


def test_offset_diagnostics_rejects_zero_cycle():
    with pytest.raises(ValidationError):
        make_diag_request(
            {
                "cycle": 0,
                "green_start": 0,
                "green_end": 10,
                "observations": [{"arrival_time": 0, "observed_green": True}],
            }
        )


def test_offset_diagnostics_rejects_cycle_over_86400():
    with pytest.raises(ValidationError):
        make_diag_request(
            {
                "cycle": 86401,
                "green_start": 0,
                "green_end": 10,
                "observations": [{"arrival_time": 0, "observed_green": True}],
            }
        )


def test_offset_diagnostics_rejects_green_start_outside_cycle():
    req = make_diag_request(
        {
            "cycle": 60,
            "green_start": 60,
            "green_end": 10,
            "observations": [{"arrival_time": 0, "observed_green": True}],
        }
    )
    with pytest.raises(HTTPException) as exc:
        req.validate_ranges()
    assert exc.value.status_code == 422


def test_offset_diagnostics_rejects_negative_green_end():
    with pytest.raises(ValidationError):
        make_diag_request(
            {
                "cycle": 60,
                "green_start": 0,
                "green_end": -1,
                "observations": [{"arrival_time": 0, "observed_green": True}],
            }
        )


def test_offset_diagnostics_rejects_empty_observations():
    with pytest.raises(ValidationError):
        make_diag_request(
            {"cycle": 60, "green_start": 0, "green_end": 10, "observations": []}
        )


def test_offset_diagnostics_rejects_too_many_observations():
    with pytest.raises(ValidationError):
        make_diag_request(
            {
                "cycle": 60,
                "green_start": 0,
                "green_end": 10,
                "observations": [
                    {"arrival_time": i, "observed_green": True}
                    for i in range(2001)
                ],
            }
        )


def test_offset_diagnostics_accepts_2000_observations_boundary():
    req = make_diag_request(
        {
            "cycle": 60,
            "green_start": 0,
            "green_end": 0,  # 常绿
            "observations": [
                {"arrival_time": i, "observed_green": True}
                for i in range(2000)
            ],
        }
    )
    req.validate_ranges()
    resp = green_wave_offset_diagnostics(req)
    assert resp.min_mismatches == 0
    assert [[w.start, w.end] for w in resp.optimal_intervals] == [[0, 60]]


def test_offset_diagnostics_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/green-wave/offset-diagnostics" in paths
    assert "post" in paths["/api/green-wave/offset-diagnostics"]


# ---------- POST /api/green-wave/slot-offsets ----------

def make_slot_request(payload):
    return SlotOffsetsRequest.model_validate(payload)


def test_slot_offsets_plain_payload():
    req = make_slot_request(
        {
            "depart_from": 0,
            "depart_to": 10,
            "adjustable_index": 0,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 30, "travel": 0}
            ],
        }
    )
    resp = green_wave_slot_offsets(req)
    assert resp.adjustable_index == 0
    assert [[w.start, w.end] for w in resp.feasible_intervals] == [[0, 21]]
    assert resp.recommended_offset == 0


def test_slot_offsets_wrapping_payload_and_recommendation():
    req = make_slot_request(
        {
            "depart_from": 0,
            "depart_to": 5,
            "adjustable_index": 0,
            "intersections": [
                {"cycle": 60, "offset": 30, "green_start": 50,
                 "green_end": 10, "travel": 0}
            ],
        }
    )
    resp = green_wave_slot_offsets(req)
    assert [[w.start, w.end] for w in resp.feasible_intervals] == [
        [0, 6], [50, 60]
    ]
    assert resp.recommended_offset == 50


def test_slot_offsets_empty_when_fixed_intersection_cannot_cover_range():
    req = make_slot_request(
        {
            "depart_from": 0,
            "depart_to": 30,
            "adjustable_index": 0,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 0, "travel": 0},
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 10, "travel": 0},
            ],
        }
    )
    resp = green_wave_slot_offsets(req)
    assert resp.feasible_intervals == []
    assert resp.recommended_offset is None


def test_slot_offsets_slack_is_honored():
    # 绿灯 [20,50)、slack=10 收缩为 [20,40)：范围 15 秒只剩 6 个可行 offset
    req = make_slot_request(
        {
            "depart_from": 0,
            "depart_to": 15,
            "adjustable_index": 0,
            "intersections": [
                {"cycle": 60, "offset": 23, "green_start": 20,
                 "green_end": 50, "travel": 0, "travel_max": 10}
            ],
        }
    )
    resp = green_wave_slot_offsets(req)
    assert [[w.start, w.end] for w in resp.feasible_intervals] == [[20, 26]]
    assert resp.recommended_offset == 23


def test_slot_offsets_rejects_index_out_of_range():
    req = make_slot_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "adjustable_index": 1,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 30, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_slot_offsets(req)
    assert exc.value.status_code == 422


def test_slot_offsets_rejects_negative_index():
    with pytest.raises(ValidationError):
        make_slot_request(
            {
                "depart_from": 0,
                "depart_to": 60,
                "adjustable_index": -1,
                "intersections": [
                    {"cycle": 60, "offset": 0, "green_start": 0,
                     "green_end": 30, "travel": 0}
                ],
            }
        )


def test_slot_offsets_reuses_green_wave_validation():
    # 与 /windows 相同的跨字段校验：路口 offset 越界与范围过长均 422。
    req = make_slot_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "adjustable_index": 0,
            "intersections": [
                {"cycle": 60, "offset": 60, "green_start": 0,
                 "green_end": 10, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_slot_offsets(req)
    assert exc.value.status_code == 422

    req2 = make_slot_request(
        {
            "depart_from": 0,
            "depart_to": 3601,
            "adjustable_index": 0,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 10, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_slot_offsets(req2)
    assert exc.value.status_code == 422


def test_slot_offsets_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/green-wave/slot-offsets" in paths
    assert "post" in paths["/api/green-wave/slot-offsets"]


# ---------- POST /api/green-wave/convoy-starts ----------

def make_convoy_request(payload):
    return ConvoyStartsRequest.model_validate(payload)


def test_convoy_starts_payload():
    # 两路口绿波带，2 辆车间隔 10 秒：首车窗口由 30 秒收缩为 20 秒
    req = make_convoy_request(
        {
            "depart_from": 0,
            "depart_to": 300,
            "vehicle_count": 2,
            "headway": 10,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30, "travel": 0},
                {"cycle": 60, "offset": 30, "green_start": 0, "green_end": 30, "travel": 30},
            ],
        }
    )
    starts = green_wave_convoy_starts(req)
    assert [[w.start, w.end] for w in starts] == [
        [0, 20], [60, 80], [120, 140], [180, 200], [240, 260],
    ]


def test_convoy_starts_empty_array_on_no_solution():
    # 3 辆车间隔 40 秒：首末车检查会误判可行，中间车必卡红灯，返回空数组
    req = make_convoy_request(
        {
            "depart_from": 0,
            "depart_to": 120,
            "vehicle_count": 3,
            "headway": 40,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30, "travel": 0}
            ],
        }
    )
    assert green_wave_convoy_starts(req) == []


def test_convoy_rejects_vehicle_count_out_of_range():
    for count in (1, 51):
        with pytest.raises(ValidationError):
            make_convoy_request(
                {
                    "depart_from": 0,
                    "depart_to": 300,
                    "vehicle_count": count,
                    "headway": 5,
                    "intersections": [
                        {"cycle": 60, "offset": 0, "green_start": 0,
                         "green_end": 30, "travel": 0}
                    ],
                }
            )


def test_convoy_rejects_non_positive_headway():
    with pytest.raises(ValidationError):
        make_convoy_request(
            {
                "depart_from": 0,
                "depart_to": 300,
                "vehicle_count": 2,
                "headway": 0,
                "intersections": [
                    {"cycle": 60, "offset": 0, "green_start": 0,
                     "green_end": 30, "travel": 0}
                ],
            }
        )


def test_convoy_rejects_span_not_less_than_range():
    # (3-1)×10 = 20 不小于范围长度 20：首车不存在可行时刻，422
    req = make_convoy_request(
        {
            "depart_from": 0,
            "depart_to": 20,
            "vehicle_count": 3,
            "headway": 10,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_convoy_starts(req)
    assert exc.value.status_code == 422


def test_convoy_span_one_less_than_range_accepted():
    # (3-1)×9 = 18 < 20：常绿路口下首车范围为 [0, 2)
    req = make_convoy_request(
        {
            "depart_from": 0,
            "depart_to": 20,
            "vehicle_count": 3,
            "headway": 9,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 0, "travel": 0}
            ],
        }
    )
    starts = green_wave_convoy_starts(req)
    assert [[w.start, w.end] for w in starts] == [[0, 2]]


def test_convoy_reuses_green_wave_validation():
    # 与 /windows 相同的跨字段校验：发车范围超过 3600 秒仍返回 422。
    req = make_convoy_request(
        {
            "depart_from": 0,
            "depart_to": 3601,
            "vehicle_count": 2,
            "headway": 5,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 10, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_convoy_starts(req)
    assert exc.value.status_code == 422


def test_convoy_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/green-wave/convoy-starts" in paths
    assert "post" in paths["/api/green-wave/convoy-starts"]


# ---------- POST /api/green-wave/timetable-reschedule ----------

def make_reschedule_request(payload):
    return TimetableRescheduleRequest.model_validate(payload)


def test_reschedule_payload_greedy_trap():
    # 贪心第一班留在 25 会让第二班无点可选；DP 整体优化为 [24,29]
    req = make_reschedule_request(
        {
            "depart_from": 0,
            "depart_to": 120,
            "original_departures": [25, 30],
            "tolerance": 5,
            "headway": 5,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 30, "travel": 0}
            ],
        }
    )
    resp = green_wave_timetable_reschedule(req)
    assert resp is not None
    assert resp.departures == [24, 29]
    assert resp.shifts == [-1, -1]
    assert resp.total_absolute_shift == 2


def test_reschedule_keeps_original_when_feasible():
    req = make_reschedule_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "original_departures": [0, 10, 20],
            "tolerance": 0,
            "headway": 5,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 30, "travel": 0}
            ],
        }
    )
    resp = green_wave_timetable_reschedule(req)
    assert resp is not None
    assert resp.departures == [0, 10, 20]
    assert resp.shifts == [0, 0, 0]
    assert resp.total_absolute_shift == 0


def test_reschedule_returns_null_when_infeasible():
    req = make_reschedule_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "original_departures": [0, 3],
            "tolerance": 0,
            "headway": 10,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 0, "travel": 0}
            ],
        }
    )
    assert green_wave_timetable_reschedule(req) is None


def test_reschedule_rejects_non_strictly_increasing_trips():
    req = make_reschedule_request(
        {
            "depart_from": 0,
            "depart_to": 60,
            "original_departures": [10, 10],
            "tolerance": 5,
            "headway": 5,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 30, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_timetable_reschedule(req)
    assert exc.value.status_code == 422


def test_reschedule_rejects_trip_outside_range():
    req = make_reschedule_request(
        {
            "depart_from": 10,
            "depart_to": 60,
            "original_departures": [5],
            "tolerance": 5,
            "headway": 5,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 30, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_timetable_reschedule(req)
    assert exc.value.status_code == 422


def test_reschedule_rejects_trip_count_out_of_range():
    base = {
        "depart_from": 0,
        "depart_to": 3000,
        "tolerance": 5,
        "headway": 5,
        "intersections": [
            {"cycle": 60, "offset": 0, "green_start": 0,
             "green_end": 0, "travel": 0}
        ],
    }
    with pytest.raises(ValidationError):
        make_reschedule_request({**base, "original_departures": []})
    with pytest.raises(ValidationError):
        make_reschedule_request(
            {**base, "original_departures": list(range(0, 255, 5))}
        )


def test_reschedule_rejects_tolerance_and_headway_ranges():
    base = {
        "depart_from": 0,
        "depart_to": 60,
        "original_departures": [10],
        "headway": 5,
        "intersections": [
            {"cycle": 60, "offset": 0, "green_start": 0,
             "green_end": 0, "travel": 0}
        ],
    }
    with pytest.raises(ValidationError):
        make_reschedule_request({**base, "tolerance": -1})
    with pytest.raises(ValidationError):
        make_reschedule_request({**base, "tolerance": 3601})
    with pytest.raises(ValidationError):
        make_reschedule_request(
            {**base, "tolerance": 5, "headway": 0}
        )
    with pytest.raises(ValidationError):
        make_reschedule_request(
            {**base, "tolerance": 5, "headway": 3601}
        )


def test_reschedule_reuses_green_wave_validation():
    # 与 /windows 相同的跨字段校验：发车范围超过 3600 秒仍返回 422。
    req = make_reschedule_request(
        {
            "depart_from": 0,
            "depart_to": 3601,
            "original_departures": [10],
            "tolerance": 5,
            "headway": 5,
            "intersections": [
                {"cycle": 60, "offset": 0, "green_start": 0,
                 "green_end": 10, "travel": 0}
            ],
        }
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_timetable_reschedule(req)
    assert exc.value.status_code == 422


def test_reschedule_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    path = "/api/green-wave/timetable-reschedule"
    assert path in paths
    assert "post" in paths[path]
    # 无解时响应体为 null（response_model 允许 None）
    schema = app.openapi()
    response = schema["paths"][path]["post"]["responses"]["200"]
    ref = response["content"]["application/json"]["schema"]
    assert {"type": "null"} in ref["anyOf"]
    assert any("$ref" in item for item in ref["anyOf"])


# ---------- POST /api/green-wave/earliest-route ----------

def make_route_request(payload):
    return EarliestRouteRequest.model_validate(payload)


def route_payload(**kw):
    base = {
        "depart_at": 0,
        "start": 0,
        "end": 2,
        "intersections": [
            {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30},
            {"cycle": 60, "offset": 0, "green_start": 20, "green_end": 50},
            {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 0},
        ],
        "roads": [
            {"from": 0, "to": 1, "travel": 10},
            {"from": 1, "to": 2, "travel": 10},
            {"from": 0, "to": 2, "travel": 35},
        ],
    }
    base.update(kw)
    return base


def test_earliest_route_payload_wait_beats_direct():
    # 0→1→2 在路口 1 等 10 秒红灯仍比直达（35 秒）更快：30 秒到达
    resp = green_wave_earliest_route(make_route_request(route_payload()))
    assert resp is not None
    assert resp.arrival == 30
    assert resp.path == [0, 1, 2]


def test_earliest_route_tie_prefers_fewer_roads():
    # 直达改为 30 秒：与 0→1→2 同时到达，道路数更少者胜
    payload = route_payload(
        roads=[
            {"from": 0, "to": 1, "travel": 10},
            {"from": 1, "to": 2, "travel": 10},
            {"from": 0, "to": 2, "travel": 30},
        ]
    )
    resp = green_wave_earliest_route(make_route_request(payload))
    assert resp is not None
    assert resp.arrival == 30
    assert resp.path == [0, 2]


def test_earliest_route_unreachable_returns_null():
    payload = route_payload(roads=[{"from": 0, "to": 1, "travel": 10}])
    assert green_wave_earliest_route(make_route_request(payload)) is None


def test_earliest_route_start_equals_end():
    payload = route_payload(start=1, end=1, depart_at=7)
    resp = green_wave_earliest_route(make_route_request(payload))
    assert resp is not None
    assert resp.arrival == 7
    assert resp.path == [1]


def test_earliest_route_rejects_intersection_count_out_of_range():
    with pytest.raises(ValidationError):
        make_route_request(
            route_payload(
                intersections=[
                    {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 0}
                ],
                roads=[],
                start=0,
                end=0,
            )
        )
    with pytest.raises(ValidationError):
        make_route_request(
            route_payload(
                intersections=[
                    {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 0}
                    for _ in range(31)
                ]
            )
        )


def test_earliest_route_rejects_too_many_roads():
    with pytest.raises(ValidationError):
        make_route_request(
            route_payload(
                roads=[
                    {"from": 0, "to": 1, "travel": 1} for _ in range(121)
                ]
            )
        )


def test_earliest_route_rejects_bad_road_fields():
    # travel 必须为正整数
    with pytest.raises(ValidationError):
        make_route_request(
            route_payload(roads=[{"from": 0, "to": 1, "travel": 0}])
        )
    # 下标不得为负
    with pytest.raises(ValidationError):
        make_route_request(
            route_payload(roads=[{"from": -1, "to": 1, "travel": 5}])
        )
    # 出发秒不得为负
    with pytest.raises(ValidationError):
        make_route_request(route_payload(depart_at=-1))


def test_earliest_route_rejects_out_of_range_indices():
    # start / end 越界（跨字段 422）
    req = make_route_request(route_payload(start=3))
    with pytest.raises(HTTPException) as exc:
        green_wave_earliest_route(req)
    assert exc.value.status_code == 422
    req = make_route_request(route_payload(end=3))
    with pytest.raises(HTTPException) as exc:
        green_wave_earliest_route(req)
    assert exc.value.status_code == 422
    # 道路端点越界（跨字段 422）
    req = make_route_request(
        route_payload(roads=[{"from": 0, "to": 3, "travel": 5}])
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_earliest_route(req)
    assert exc.value.status_code == 422


def test_earliest_route_rejects_phase_out_of_range():
    req = make_route_request(
        route_payload(
            intersections=[
                {"cycle": 60, "offset": 60, "green_start": 0, "green_end": 30},
                {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 0},
            ]
        )
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_earliest_route(req)
    assert exc.value.status_code == 422


def test_earliest_route_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    path = "/api/green-wave/earliest-route"
    assert path in paths
    assert "post" in paths[path]
    # 不可达时响应体为 null（response_model 允许 None）
    schema = app.openapi()
    response = schema["paths"][path]["post"]["responses"]["200"]
    ref = response["content"]["application/json"]["schema"]
    assert {"type": "null"} in ref["anyOf"]
    assert any("$ref" in item for item in ref["anyOf"])


# ---------- POST /api/green-wave/queue-forecast ----------

def make_queue_request(payload):
    return QueueForecastRequest.model_validate(payload)


def queue_payload(**kw):
    base = {
        "forecast_from": 0,
        "forecast_to": 60,
        "cycle": 60,
        "offset": 0,
        "green_start": 0,
        "green_end": 0,  # 常绿
        "discharge_per_second": 1,
        "initial_queue": 0,
        "arrivals": [],
    }
    base.update(kw)
    return base


def test_queue_forecast_payload():
    # 常绿每秒放行 1：期初 2 辆先走，t=1 到达的 3 辆批次在 t=4 走完
    req = make_queue_request(
        queue_payload(initial_queue=2, arrivals=[{"time": 1, "vehicles": 3}])
    )
    resp = green_wave_queue_forecast(req)
    assert resp.batch_pass_times == [4]
    assert resp.final_queue == 0
    assert resp.max_queue == 4


def test_queue_forecast_red_light_and_null_pass_time():
    # 绿灯 [0,10)：t=20 到达的批次在范围内一直遇红，全部待行
    req = make_queue_request(
        queue_payload(
            green_start=0,
            green_end=10,
            discharge_per_second=5,
            arrivals=[{"time": 20, "vehicles": 3}],
        )
    )
    resp = green_wave_queue_forecast(req)
    assert resp.batch_pass_times == [None]
    assert resp.final_queue == 3
    assert resp.max_queue == 3


def test_queue_forecast_same_second_batches_input_order():
    req = make_queue_request(
        queue_payload(
            forecast_to=10,
            arrivals=[
                {"time": 0, "vehicles": 2},
                {"time": 0, "vehicles": 1},
            ],
        )
    )
    resp = green_wave_queue_forecast(req)
    assert resp.batch_pass_times == [1, 2]
    assert resp.final_queue == 0
    assert resp.max_queue == 3


def test_queue_forecast_rejects_arrival_outside_range():
    req = make_queue_request(
        queue_payload(arrivals=[{"time": 60, "vehicles": 1}])
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_queue_forecast(req)
    assert exc.value.status_code == 422
    req = make_queue_request(
        queue_payload(arrivals=[{"time": -1, "vehicles": 1}])
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_queue_forecast(req)
    assert exc.value.status_code == 422


def test_queue_forecast_rejects_invalid_signal_params():
    # offset / green_start / green_end 必须落在 [0, cycle) 内
    for field, value in (("offset", 60), ("green_start", 60), ("green_end", 60)):
        req = make_queue_request(queue_payload(**{field: value}))
        with pytest.raises(HTTPException) as exc:
            green_wave_queue_forecast(req)
        assert exc.value.status_code == 422
    with pytest.raises(ValidationError):
        make_queue_request(queue_payload(cycle=0))


def test_queue_forecast_rejects_bad_range():
    # 空范围与超长范围均 422
    req = make_queue_request(queue_payload(forecast_from=10, forecast_to=10))
    with pytest.raises(HTTPException) as exc:
        green_wave_queue_forecast(req)
    assert exc.value.status_code == 422
    req = make_queue_request(queue_payload(forecast_to=86401))
    with pytest.raises(HTTPException) as exc:
        green_wave_queue_forecast(req)
    assert exc.value.status_code == 422


def test_queue_forecast_rejects_bad_batch_fields():
    # 车辆数必须为正整数
    with pytest.raises(ValidationError):
        make_queue_request(
            queue_payload(arrivals=[{"time": 0, "vehicles": 0}])
        )
    # 批次数超过 1000
    with pytest.raises(ValidationError):
        make_queue_request(
            queue_payload(
                forecast_to=2000,
                arrivals=[
                    {"time": i, "vehicles": 1} for i in range(1001)
                ],
            )
        )
    # 放行量与期初队列的字段级约束
    with pytest.raises(ValidationError):
        make_queue_request(queue_payload(discharge_per_second=0))
    with pytest.raises(ValidationError):
        make_queue_request(queue_payload(initial_queue=-1))


def test_queue_forecast_accepts_1000_batches_boundary():
    req = make_queue_request(
        queue_payload(
            forecast_to=86400,
            arrivals=[
                {"time": 80 * i, "vehicles": 3} for i in range(1000)
            ],
        )
    )
    resp = green_wave_queue_forecast(req)
    assert resp.batch_pass_times == [80 * i + 2 for i in range(1000)]
    assert resp.final_queue == 0
    assert resp.max_queue == 3


def test_queue_forecast_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/green-wave/queue-forecast" in paths
    assert "post" in paths["/api/green-wave/queue-forecast"]


# ---------- POST /api/green-wave/min-discharge ----------


def make_min_discharge_request(payload):
    return MinDischargeRequest.model_validate(payload)


def min_discharge_payload(**kw):
    base = {
        "forecast_from": 0,
        "forecast_to": 60,
        "cycle": 60,
        "offset": 0,
        "green_start": 0,
        "green_end": 0,  # 常绿
        "initial_queue": 0,
        "arrivals": [],
    }
    base.update(kw)
    return base


def test_min_discharge_payload():
    # 常绿：t=0 到 3 辆、deadline=4，d=1 时 t=0..2 走完；返回最低放行量 1
    req = make_min_discharge_request(
        min_discharge_payload(
            forecast_to=5,
            arrivals=[{"time": 0, "vehicles": 3, "deadline": 4}],
        )
    )
    resp = green_wave_min_discharge(req)
    assert resp is not None
    assert resp.min_discharge_per_second == 1
    assert resp.batch_pass_times == [2]


def test_min_discharge_tight_deadline_requires_capacity():
    # t=0 到 4 辆、deadline=1：d=2 时 t=0、1 恰走完（含端点）
    req = make_min_discharge_request(
        min_discharge_payload(
            arrivals=[{"time": 0, "vehicles": 4, "deadline": 1}],
        )
    )
    resp = green_wave_min_discharge(req)
    assert resp is not None
    assert resp.min_discharge_per_second == 2
    assert resp.batch_pass_times == [1]


def test_min_discharge_same_second_batches_input_order():
    req = make_min_discharge_request(
        min_discharge_payload(
            forecast_to=10,
            arrivals=[
                {"time": 0, "vehicles": 2, "deadline": 9},
                {"time": 0, "vehicles": 1, "deadline": 1},
            ],
        )
    )
    resp = green_wave_min_discharge(req)
    # d=2：第一批 t=0 走完、第二批 t=1 走完（恰为 deadline）
    assert resp is not None
    assert resp.min_discharge_per_second == 2
    assert resp.batch_pass_times == [0, 1]


def test_min_discharge_initial_queue_and_red_phase():
    # 绿灯 [0,10)：期初 5 辆，t=0 起 10 个绿灯秒内须清空；d=1 即可
    req = make_min_discharge_request(
        min_discharge_payload(
            green_start=0,
            green_end=10,
            initial_queue=5,
            arrivals=[],
        )
    )
    resp = green_wave_min_discharge(req)
    assert resp is not None
    assert resp.min_discharge_per_second == 1
    assert resp.batch_pass_times == []
    # t=20 到达的批次在范围 [0,60) 内一直遇红：无解 null
    req = make_min_discharge_request(
        min_discharge_payload(
            green_start=0,
            green_end=10,
            arrivals=[{"time": 20, "vehicles": 1, "deadline": 59}],
        )
    )
    assert green_wave_min_discharge(req) is None


def test_min_discharge_no_vehicles_returns_one():
    req = make_min_discharge_request(min_discharge_payload())
    resp = green_wave_min_discharge(req)
    assert resp is not None
    assert resp.min_discharge_per_second == 1
    assert resp.batch_pass_times == []


def test_min_discharge_rejects_deadline_before_arrival():
    req = make_min_discharge_request(
        min_discharge_payload(
            arrivals=[{"time": 10, "vehicles": 1, "deadline": 9}],
        )
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_min_discharge(req)
    assert exc.value.status_code == 422


def test_min_discharge_rejects_deadline_at_range_end():
    # deadline 必须严格小于 forecast_to（范围终点不含）
    req = make_min_discharge_request(
        min_discharge_payload(
            arrivals=[{"time": 0, "vehicles": 1, "deadline": 60}],
        )
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_min_discharge(req)
    assert exc.value.status_code == 422


def test_min_discharge_rejects_arrival_outside_range():
    req = make_min_discharge_request(
        min_discharge_payload(
            arrivals=[{"time": 60, "vehicles": 1, "deadline": 60}],
        )
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_min_discharge(req)
    assert exc.value.status_code == 422


def test_min_discharge_rejects_invalid_signal_and_range():
    for field, value in (("offset", 60), ("green_start", 60), ("green_end", 60)):
        req = make_min_discharge_request(min_discharge_payload(**{field: value}))
        with pytest.raises(HTTPException) as exc:
            green_wave_min_discharge(req)
        assert exc.value.status_code == 422
    req = make_min_discharge_request(
        min_discharge_payload(forecast_from=10, forecast_to=10)
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_min_discharge(req)
    assert exc.value.status_code == 422
    req = make_min_discharge_request(min_discharge_payload(forecast_to=86401))
    with pytest.raises(HTTPException) as exc:
        green_wave_min_discharge(req)
    assert exc.value.status_code == 422


def test_min_discharge_rejects_bad_batch_fields():
    with pytest.raises(ValidationError):
        make_min_discharge_request(
            min_discharge_payload(
                arrivals=[{"time": 0, "vehicles": 0, "deadline": 1}]
            )
        )
    with pytest.raises(ValidationError):
        make_min_discharge_request(
            min_discharge_payload(
                forecast_to=2000,
                arrivals=[
                    {"time": i, "vehicles": 1, "deadline": i}
                    for i in range(1001)
                ],
            )
        )
    with pytest.raises(ValidationError):
        make_min_discharge_request(min_discharge_payload(initial_queue=-1))


def test_min_discharge_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    path = "/api/green-wave/min-discharge"
    assert path in paths
    assert "post" in paths[path]
    # 无解时响应体为 null（response_model 允许 None）
    schema = app.openapi()
    response = schema["paths"][path]["post"]["responses"]["200"]
    ref = response["content"]["application/json"]["schema"]
    assert {"type": "null"} in ref["anyOf"]
    assert any("$ref" in item for item in ref["anyOf"])


# ---------- POST /api/green-wave/conflict-audit ----------


def make_conflict_audit_request(payload):
    return ConflictAuditRequest.model_validate(payload)


def conflict_payload(**kw):
    base = {
        "audit_from": 0,
        "audit_to": 120,
        "streams": [
            {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30},
            {"cycle": 60, "offset": 0, "green_start": 20, "green_end": 50},
        ],
        "pairs": [{"a": 0, "b": 1, "clearance": 5}],
    }
    base.update(kw)
    return base


def test_conflict_audit_payload_overlap_and_tail():
    # A=[0,30)、B=[20,50)、c=5：重叠 [20,30) 加 A 尾部 [30,35) 合并为 [20,35)
    req = make_conflict_audit_request(conflict_payload())
    resp = green_wave_conflict_audit(req)
    assert resp.audit_from == 0
    assert resp.audit_to == 120
    assert len(resp.pairs) == 1
    pair = resp.pairs[0]
    assert pair.a == 0
    assert pair.b == 1
    assert pair.clearance == 5
    assert [[w.start, w.end] for w in pair.danger_intervals] == [
        [20, 35], [80, 95],
    ]


def test_conflict_audit_safe_pair_returns_empty_intervals():
    # 首尾相接、c=0：无同时绿灯，安全
    payload = conflict_payload(
        streams=[
            {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30},
            {"cycle": 60, "offset": 0, "green_start": 30, "green_end": 50},
        ],
        pairs=[{"a": 0, "b": 1, "clearance": 0}],
    )
    resp = green_wave_conflict_audit(make_conflict_audit_request(payload))
    assert resp.pairs[0].danger_intervals == []


def test_conflict_audit_green_before_range_contributes_tail():
    # 范围从 32 开始：范围前结束的 A 绿灯尾部仍裁剪进 [32,40)
    payload = conflict_payload(
        audit_from=32,
        streams=[
            {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30},
            {"cycle": 60, "offset": 0, "green_start": 30, "green_end": 50},
        ],
        pairs=[{"a": 0, "b": 1, "clearance": 10}],
    )
    resp = green_wave_conflict_audit(make_conflict_audit_request(payload))
    assert [[w.start, w.end] for w in resp.pairs[0].danger_intervals] == [
        [32, 40], [90, 100],
    ]


def test_conflict_audit_multiple_pairs_input_order():
    payload = conflict_payload(
        streams=[
            {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30},
            {"cycle": 60, "offset": 0, "green_start": 20, "green_end": 50},
            {"cycle": 60, "offset": 0, "green_start": 40, "green_end": 55},
        ],
        pairs=[
            {"a": 2, "b": 0, "clearance": 0},
            {"a": 0, "b": 1, "clearance": 5},
        ],
    )
    resp = green_wave_conflict_audit(make_conflict_audit_request(payload))
    assert [(p.a, p.b) for p in resp.pairs] == [(2, 0), (0, 1)]
    # 第一对：A=[0,30) 与 C=[40,55) c=0 从不重叠
    assert resp.pairs[0].danger_intervals == []
    assert [[w.start, w.end] for w in resp.pairs[1].danger_intervals] == [
        [20, 35], [80, 95],
    ]


def test_conflict_audit_empty_pairs_accepted():
    payload = conflict_payload(pairs=[])
    resp = green_wave_conflict_audit(make_conflict_audit_request(payload))
    assert resp.pairs == []


def test_conflict_audit_rejects_stream_count_out_of_range():
    with pytest.raises(ValidationError):
        make_conflict_audit_request(
            conflict_payload(
                streams=[
                    {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30}
                ]
            )
        )
    with pytest.raises(ValidationError):
        make_conflict_audit_request(
            conflict_payload(
                streams=[
                    {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30}
                    for _ in range(9)
                ],
                pairs=[],
            )
        )


def test_conflict_audit_rejects_clearance_out_of_range():
    with pytest.raises(ValidationError):
        make_conflict_audit_request(
            conflict_payload(pairs=[{"a": 0, "b": 1, "clearance": -1}])
        )
    with pytest.raises(ValidationError):
        make_conflict_audit_request(
            conflict_payload(pairs=[{"a": 0, "b": 1, "clearance": 301}])
        )


def test_conflict_audit_rejects_phase_out_of_range():
    # offset / green_start / green_end 必须落在 [0, cycle) 内（第二股车流）
    for field, value in (("offset", 60), ("green_start", 60), ("green_end", 60)):
        req = make_conflict_audit_request(
            conflict_payload(
                streams=[
                    {"cycle": 60, "offset": 0, "green_start": 0, "green_end": 30},
                    {"cycle": 60, "offset": 0, "green_start": 20,
                     "green_end": 50, **({field: value})},
                ]
            )
        )
        with pytest.raises(HTTPException) as exc:
            green_wave_conflict_audit(req)
        assert exc.value.status_code == 422
    with pytest.raises(ValidationError):
        make_conflict_audit_request(
            conflict_payload(
                streams=[
                    {"cycle": 0, "offset": 0, "green_start": 0, "green_end": 0},
                    {"cycle": 60, "offset": 0, "green_start": 20, "green_end": 50},
                ]
            )
        )


def test_conflict_audit_rejects_bad_range():
    # 空范围与超长范围均 422
    req = make_conflict_audit_request(
        conflict_payload(audit_from=10, audit_to=10)
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_conflict_audit(req)
    assert exc.value.status_code == 422
    req = make_conflict_audit_request(
        conflict_payload(audit_from=0, audit_to=86401)
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_conflict_audit(req)
    assert exc.value.status_code == 422


def test_conflict_audit_rejects_pair_index_out_of_range():
    req = make_conflict_audit_request(
        conflict_payload(pairs=[{"a": 0, "b": 2, "clearance": 5}])
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_conflict_audit(req)
    assert exc.value.status_code == 422


def test_conflict_audit_rejects_same_stream_pair():
    req = make_conflict_audit_request(
        conflict_payload(pairs=[{"a": 1, "b": 1, "clearance": 5}])
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_conflict_audit(req)
    assert exc.value.status_code == 422


def test_conflict_audit_rejects_duplicate_pair():
    # (0,1) 与 (1,0) 是同一个无序对
    req = make_conflict_audit_request(
        conflict_payload(
            pairs=[
                {"a": 0, "b": 1, "clearance": 5},
                {"a": 1, "b": 0, "clearance": 3},
            ]
        )
    )
    with pytest.raises(HTTPException) as exc:
        green_wave_conflict_audit(req)
    assert exc.value.status_code == 422


def test_conflict_audit_accepts_full_day_range_boundary():
    payload = conflict_payload(
        audit_from=0,
        audit_to=86400,
        pairs=[{"a": 0, "b": 1, "clearance": 300}],
    )
    req = make_conflict_audit_request(payload)
    req.validate_ranges()
    resp = green_wave_conflict_audit(req)
    # 周期 60：每个周期危险 [20,35)，全天 86400/60=1440 段
    assert len(resp.pairs[0].danger_intervals) == 1440


def test_conflict_audit_route_is_in_openapi():
    from app.main import app

    paths = app.openapi()["paths"]
    path = "/api/green-wave/conflict-audit"
    assert path in paths
    assert "post" in paths[path]
