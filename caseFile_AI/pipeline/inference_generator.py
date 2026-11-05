"""
Temperature and relative-humidity inference generator.

Accepts the combined rht_results/summary.json or either standalone
predictions_summary.json / predictions_RH_summary.json file.

Usage:
    python inference_generator.py rht_results/summary.json
"""

import json
import math
import sys


def _number(data: dict, key: str):
    value = data.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _prediction_record(item: dict) -> dict:
    temperature = item.get("T")
    humidity = item.get("RH")
    t_min = t_max = rh_min = rh_max = None

    if isinstance(temperature, dict):
        t_min = _number(temperature, "min_c")
        t_max = _number(temperature, "max_c")
        if t_min is None:
            value = _number(temperature, "min_k")
            t_min = value - 273.15 if value is not None else None
        if t_max is None:
            value = _number(temperature, "max_k")
            t_max = value - 273.15 if value is not None else None

    if isinstance(humidity, dict):
        rh_min = _number(humidity, "min_percent")
        rh_max = _number(humidity, "max_percent")

    if all(value is None for value in (t_min, t_max, rh_min, rh_max)):
        return {}

    return {
        "time_label": (
            item.get("time_label") or item.get("label") or item.get("clock")
            or item.get("timestep") or "unknown time"
        ),
        "t_min": t_min,
        "t_max": t_max,
        "rh_min": rh_min,
        "rh_max": rh_max,
    }


def _humidity_description(rh_min: float, rh_max: float) -> str:
    if rh_max < 30:
        return "the mapped area is generally dry"
    if rh_min >= 60:
        return "the mapped area is generally humid"
    if rh_min < 30 <= rh_max:
        upper = "humid" if rh_max >= 60 else "moderate"
        return f"relative humidity varies from dry to {upper} conditions"
    return "relative humidity is mostly in a moderate range"


def _format_record(record: dict) -> str:
    observations = []
    if record["t_min"] is not None and record["t_max"] is not None:
        observations.append(
            f"predicted air temperature ranges from {record['t_min']:.1f} °C "
            f"to {record['t_max']:.1f} °C"
        )
    if record["rh_min"] is not None and record["rh_max"] is not None:
        observations.append(
            f"relative humidity ranges from {record['rh_min']:.1f}% "
            f"to {record['rh_max']:.1f}% "
            f"({_humidity_description(record['rh_min'], record['rh_max'])})"
        )
    return f"At {record['time_label']}: " + "; ".join(observations) + "."


def _single_report(record: dict) -> str:
    return (
        _format_record(record)
        + "\n\nThese are raster minimum and maximum values, not spatial averages."
    )


def generate_report(summary: dict) -> str:
    """Create per-timestep and overall T/RH inferences from prediction JSON."""
    raw_times = summary.get("times")
    if isinstance(raw_times, list):
        records = [
            record for item in raw_times if isinstance(item, dict)
            if (record := _prediction_record(item))
        ]
        if records:
            sections = [_format_record(record) for record in records]
            temperatures = [
                record[key] for record in records for key in ("t_min", "t_max")
                if record[key] is not None
            ]
            humidities = [
                record[key] for record in records for key in ("rh_min", "rh_max")
                if record[key] is not None
            ]
            overall = []
            if temperatures:
                overall.append(
                    f"temperature {min(temperatures):.1f}–{max(temperatures):.1f} °C"
                )
            if humidities:
                overall.append(
                    f"relative humidity {min(humidities):.1f}–{max(humidities):.1f}%"
                )
            sections.append(
                "Overall predicted spatial ranges across the reported timesteps: "
                + "; ".join(overall) + "."
            )
            sections.append(
                "These are raster minimum and maximum values, not spatial averages."
            )
            return "\n\n".join(sections)

    matched = summary.get("matched")
    if isinstance(matched, dict):
        record = _prediction_record({
            "time_label": matched.get("clock") or matched.get("timestep", "unknown time"),
            "T": {"min_k": matched.get("T_min"), "max_k": matched.get("T_max")},
        })
        if record:
            return _single_report(record)

    rh_min = _number(summary, "RH_min_percent")
    rh_max = _number(summary, "RH_max_percent")
    if rh_min is not None and rh_max is not None:
        return _single_report({
            "time_label": summary.get("clock") or summary.get("timestep", "unknown time"),
            "t_min": None,
            "t_max": None,
            "rh_min": rh_min,
            "rh_max": rh_max,
        })

    return "No valid temperature or relative-humidity prediction data found in the input JSON."


DEFAULT_SUMMARY_FILE = "summary.json"

if __name__ == "__main__":
    summary_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUMMARY_FILE
    with open(summary_path, "r", encoding="utf-8") as summary_file:
        print(generate_report(json.load(summary_file)))
