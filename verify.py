# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""NPU acceptance driver. Result files are measured, never copied from fixtures."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable

from paired_statistics import summarize

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "fixtures"))
import generate_data
import golden

_I32 = 2147483647


def _inputs(case: dict) -> tuple[list, dict, list]:
    values, attrs = generate_data.generate(case)
    tensors = [torch.tensor(value, dtype=torch.int32, device="npu:0") for value in values]
    if case.get("offset"):
        tensors = [_offset_copy(tensor) for tensor in tensors]
    return tensors, attrs, values


def _offset_copy(tensor: "torch.Tensor") -> "torch.Tensor":
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    view = storage[1:].view(tensor.shape)
    view.copy_(tensor)
    return view


def _info(tensors: list) -> dict:
    mapping = tensors[1]
    return {"layers": 1 if mapping.ndim == 1 else mapping.shape[0],
            "blocks": mapping.shape[-1], "single": mapping.ndim == 1}


def _oracle(tensors: list, attrs: dict) -> "torch.Tensor":
    values = [tensor.cpu().tolist() for tensor in tensors]
    info = _info(tensors)
    shape = (tensors[0].numel(),) if info["single"] else (info["layers"], tensors[0].numel())
    return torch.tensor(golden.calculate(values, attrs, info["single"]),
                        dtype=torch.int32).reshape(shape)


def _check_output(output: "torch.Tensor", expected: "torch.Tensor", tensors: list) -> dict:
    checks = {"shape": tuple(output.shape) == tuple(expected.shape),
              "dtype": output.dtype == torch.int32, "npu": output.device == tensors[0].device,
              "contiguous": output.is_contiguous(),
              "independent_storage": all(output.numel() == 0 or output.untyped_storage().data_ptr()
                                         != tensor.untyped_storage().data_ptr() for tensor in tensors)}
    checks["candidate_mismatches"] = int((output.cpu() != expected).sum().item())
    if not all(checks[key] for key in checks if key != "candidate_mismatches") or checks["candidate_mismatches"]:
        raise AssertionError(f"output checks failed: {checks}")
    return checks


def _case(case: dict) -> dict:
    tensors, attrs, _ = _inputs(case)
    originals = [tensor.clone() for tensor in tensors]
    prepared = solution.prepare(tensors, attrs)
    expected = _oracle(tensors, attrs)
    output, retained = prepared.run(), prepared.run()
    checks = _check_output(output, expected, tensors)
    _check_output(retained, expected, tensors)
    if output is retained or (output.numel() and output.untyped_storage().data_ptr() == retained.untyped_storage().data_ptr()):
        raise AssertionError("successive outputs share storage")
    baseline = golden.reference("baseline").prepare(tensors, attrs, _info(tensors))
    baseline_output, baseline_retained = baseline(), baseline()
    baseline_checks = _check_output(baseline_output, expected, tensors)
    _check_output(baseline_retained, expected, tensors)
    if baseline_output.numel() and baseline_output.untyped_storage().data_ptr() == baseline_retained.untyped_storage().data_ptr():
        raise AssertionError("baseline successive outputs share storage")
    checks["baseline_mismatches"] = baseline_checks["candidate_mismatches"]
    checks["inputs_unchanged"] = all(torch.equal(x, y) for x, y in zip(tensors, originals))
    if checks["baseline_mismatches"] or not checks["inputs_unchanged"]:
        raise AssertionError(checks)
    expected_hash = case.get("generalization", {}).get("expected_sha256")
    actual_hash = hashlib.sha256(json.dumps(expected.tolist(), separators=(",", ":")).encode()).hexdigest()
    if expected_hash is not None and actual_hash != expected_hash:
        raise AssertionError("generated oracle differs from published generalization hash")
    return {"id": case["id"], "status": "PASS", "checks": checks, "golden_sha256": actual_hash,
            "candidate_mismatches": checks["candidate_mismatches"],
            "baseline_mismatches": checks["baseline_mismatches"],
            "shape": list(output.shape), "input_shapes": [list(t.shape) for t in tensors],
            "attributes": attrs, "storage_offsets": [t.storage_offset() for t in tensors]}


def _negative(case: dict) -> list[dict]:
    ids = ("input_count", "unknown_attribute", "cpu_input", "dtype", "noncontiguous",
           "negative_mapping", "logical_address_out_of_range", "zero_block_size",
           "bool_block_size", "output_int32_overflow")
    results = []
    for name in ids:
        tensors, attrs, _ = _inputs(case)
        if name == "input_count":
            tensors.pop()
        elif name == "unknown_attribute":
            attrs["unknown"] = 1
        elif name == "cpu_input":
            tensors[0] = tensors[0].cpu()
        elif name == "dtype":
            tensors[0] = tensors[0].to(torch.int64)
        elif name == "noncontiguous":
            tensors[0] = torch.zeros(tensors[0].numel() * 2, dtype=torch.int32, device="npu:0")[::2]
        elif name == "negative_mapping":
            tensors[1].view(-1)[0] = -1
        elif name == "logical_address_out_of_range":
            tensors[0][0] = _I32
        elif name == "zero_block_size":
            attrs["block_size"] = 0
        elif name == "bool_block_size":
            attrs["block_size"] = True
        elif name == "output_int32_overflow":
            tensors[1].view(-1)[0] = _I32
        try:
            solution.prepare(tensors, attrs)
        except (TypeError, ValueError) as error:
            results.append({"id": name, "status": "PASS", "exception": type(error).__name__})
        else:
            raise AssertionError(f"prepare accepted {name}")
    return results


def _updates(case: dict) -> tuple[list, list]:
    tensors, attrs, _ = _inputs(case)
    prepared = solution.prepare(tensors, attrs)
    dynamic, rebindings = [], []
    tensors[0].copy_(tensors[0].roll(1))
    checks = _check_output(prepared.run(), _oracle(tensors, attrs), tensors)
    dynamic.append({"id": "logical_values_changed", "status": "PASS", **checks})
    tensors[1].copy_(tensors[1].flip([-1]))
    checks = _check_output(prepared.run(), _oracle(tensors, attrs), tensors)
    dynamic.append({"id": "mapping_values_changed", "status": "PASS", **checks})
    if case["id"] == "slot_perf_large":
        tensors[1].fill_((_I32 - (attrs["block_size"] - 1)) // attrs["block_size"])
        tensors[0][1] = attrs["block_size"] - 1
        checks = _check_output(prepared.run(), _oracle(tensors, attrs), tensors)
        dynamic.append({"id": "large_path_int32_output_boundary", "status": "PASS", **checks})
    for index, name in ((1, "mapping_set_storage"), (0, "logical_set_storage")):
        replacement = _offset_copy(tensors[index].roll(1, dims=-1))
        old_pointer = tensors[index].data_ptr()
        tensors[index].set_(replacement)
        checks = _check_output(prepared.run(), _oracle(tensors, attrs), tensors)
        pointer = tensors[index].data_ptr()
        if old_pointer == pointer or tensors[index].storage_offset() != 1:
            raise AssertionError("storage rebind precondition failed")
        rebindings.append({"id": name, "status": "PASS", **checks, "storage_offset": 1,
                           "pointer_changed": True, "old_alignment_mod16": old_pointer % 16,
                           "new_alignment_mod16": pointer % 16})
    return dynamic, rebindings


def _allocation(call: Callable, output_bytes: int) -> dict:
    for _ in range(5):
        output = call()
        del output
    torch.npu.synchronize()
    gc.collect()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    output = call()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - before
    if output.numel() * output.element_size() != output_bytes:
        raise AssertionError("output size differs from contract")
    del output
    for _ in range(20):
        output = call()
        del output
    torch.npu.synchronize()
    gc.collect()
    after = torch.npu.memory_allocated()
    if before != after:
        raise AssertionError(f"active allocation did not recover: {before} -> {after}")
    return {"active_before_bytes": before, "active_after_bytes": after,
            "peak_increment_bytes": peak, "output_bytes": output_bytes,
            "repeat20_released": "PASS"}


def _streams(tensors: list, attrs: dict, baseline: bool = False) -> None:
    retained, references = [], []
    for shift in (1, 2):
        inputs = [tensor.clone() for tensor in tensors]
        prepared = (golden.reference("baseline").prepare(inputs, attrs, _info(inputs))
                    if baseline else solution.prepare(inputs, attrs).run)
        expected_values = [inputs[0].cpu().roll(shift).tolist(), inputs[1].cpu().tolist()]
        expected = torch.tensor(golden.calculate(expected_values, attrs, _info(inputs)["single"]),
                                dtype=torch.int32)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            inputs[0].copy_(inputs[0].roll(shift))
            outputs = [prepared() for _ in range(3)]
        retained.extend((output, expected) for output in outputs)
        references.append((inputs, prepared, stream))
    torch.npu.synchronize()
    pointers = [output.untyped_storage().data_ptr() for output, _ in retained]
    if len(set(pointers)) != 6:
        raise AssertionError("retained outputs alias across streams")
    for output, expected in retained:
        if not torch.equal(output.cpu(), expected):
            raise AssertionError("stream-dependent input or retained output mismatch")


def _engineering(case: dict) -> dict:
    tensors, attrs, _ = _inputs(case)
    prepared = solution.prepare(tensors, attrs)
    baseline = golden.reference("baseline").prepare(tensors, attrs, _info(tensors))
    _streams(tensors, attrs)
    _streams(tensors, attrs, baseline=True)
    size = _info(tensors)["layers"] * tensors[0].numel() * 4
    candidate_memory = _allocation(prepared.run, size)
    baseline_memory = _allocation(baseline, size)
    return {**candidate_memory, "two_streams": "PASS", "retained_outputs_independent": "PASS",
            "baseline_two_streams": "PASS", "baseline_retained_outputs_independent": "PASS",
            "baseline_peak_increment_bytes": baseline_memory["peak_increment_bytes"],
            "baseline_output_bytes": size,
            "baseline_active_before_bytes": baseline_memory["active_before_bytes"],
            "baseline_active_after_bytes": baseline_memory["active_after_bytes"],
            "candidate_minus_baseline_peak_bytes": candidate_memory["peak_increment_bytes"]
                - baseline_memory["peak_increment_bytes"],
            "peak_definition": "Warmed active PyTorch allocation increment for one complete call"}


def _supplemental() -> dict:
    nonpower, powers = [], []
    for size in (3, 7, 31):
        maximum = (_I32 - (size - 1)) // size
        case = {"id": f"large_nonpower_{size}_near_i32",
                "values": [[-1, 0, size - 1, size, 4096 * size - 1] * 13107 + [-7],
                           [[maximum - (index % 3) for index in range(4096)] for _ in range(32)]],
                "attributes": {"block_size": size}}
        nonpower.append(_case(case))
    other = {}
    for size, suffix in ((1073741823, "nonpower"), (1073741824, "power2")):
        pattern = [-2147483648, -1, 0, size - 1, size, 2 * size - 1, _I32]
        for large in (False, True):
            name = f"{'large' if large else 'small'}_logical_i32_{suffix}"
            values = [(pattern * 147)[:1024], [[0, 1, 0] for _ in range(32)]] if large else [pattern, [0, 1, 0]]
            result = _case({"id": name, "values": values, "attributes": {"block_size": size}})
            if suffix == "power2":
                powers.append(result)
            else:
                other[f"{'large' if large else 'small'}_logical_i32_boundary"] = result
    return {"large_nonpower_boundaries": nonpower, "power2_logical_i32_boundaries": powers, **other}


def _benchmark(case: dict, partial: dict) -> dict:
    tensors, attrs, _ = _inputs(case)
    prepared = solution.prepare(tensors, attrs)
    calls = {"candidate": prepared.run,
             "baseline": golden.reference("baseline").prepare(tensors, attrs, _info(tensors))}
    for call in calls.values():
        for _ in range(50):
            output = call()
            del output
        torch.npu.synchronize()
    rounds = {"baseline": [], "candidate": []}
    orders, elapsed_ns = [], []
    partial.update({"baseline_rounds_us": rounds["baseline"],
                    "candidate_rounds_us": rounds["candidate"],
                    "round_order": orders, "elapsed_ns": elapsed_ns})
    for index in range(6):
        order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        raw = {}
        orders.append(order)
        elapsed_ns.append(raw)
        for name in order:
            call = calls[name]
            torch.npu.synchronize()
            begin = time.perf_counter_ns()
            for _ in range(200):
                output = call()
                del output
            torch.npu.synchronize()
            duration = time.perf_counter_ns() - begin
            rounds[name].append(duration / 200 / 1000)
            raw[name] = duration
    target = {"slot_perf_small": 85, "slot_perf_large": 155}.get(case["id"])
    summary = summarize(rounds["baseline"], rounds["candidate"], target)
    # Route diagnostics have no improvement/performance gate.
    summary["status"] = summary["status"] if target is not None else "MEASURED"
    return {"id": case["id"], **summary, "round_order": orders,
            "elapsed_ns": elapsed_ns, "warmup": 50, "iterations": 200,
            "measurement_protocol": "complete prepared.run, wall time, final device synchronization"}


def _assert_coverage(result: dict, generalization: dict, contract: dict) -> None:
    actual_ids = {record["id"] for record in result["cases"] + generalization["cases"]}
    if actual_ids != {case["id"] for case in contract["input_cases"]}:
        raise AssertionError("positive input coverage is incomplete")
    expected = contract["expected_checks"]["result.json"]
    if {record["id"] for record in result["negative"]} != {record["id"] for record in expected["negative"]}:
        raise AssertionError("negative input coverage is incomplete")
    for group in ("dynamic_updates", "storage_rebindings"):
        for name, records in expected[group].items():
            if {record["id"] for record in result[group][name]} != {record["id"] for record in records}:
                raise AssertionError(f"incomplete {group}: {name}")
    if set(result["engineering_by_case"]) != set(expected["engineering_by_case"]):
        raise AssertionError("engineering coverage is incomplete")
    for group in ("large_nonpower_boundaries", "power2_logical_i32_boundaries"):
        if {record["id"] for record in result[group]} != {record["id"] for record in expected[group]}:
            raise AssertionError(f"incomplete boundary coverage: {group}")
    for name in ("large_logical_i32_boundary", "small_logical_i32_boundary"):
        if result[name]["id"] != expected[name]["id"]:
            raise AssertionError(f"missing {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--functional-only", action="store_true")
    parser.add_argument("--typical", action="store_true", help="fixed tuning case: slot_perf_large")
    parser.add_argument("--confirm-task-timing", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == _ROOT or _ROOT in output.parents:
        parser.error("results must be outside the submitted component")
    if not args.functional_only and not args.confirm_task_timing:
        parser.error("confirm attachment 50/6/200 timing before measurement")
    if args.functional_only and args.typical:
        parser.error("typical is a performance phase")
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "RUNNING", "cases": [], "benchmarks": []}
    source_files = sorted(_ROOT.glob("*.py")) + sorted((_ROOT / "fixtures").glob("*.py"))
    source_files += sorted((_ROOT / "fixtures").glob("*.json"))
    result["source_sha256"] = {str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in source_files}
    version = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False)
    result["tested_commit"] = version.stdout.strip() if version.returncode == 0 else None
    generalization = {"cases": [], "benchmarks": []}
    exit_code = 1
    try:
        global torch, solution
        import torch
        import torch_npu
        import triton
        import solution

        if not torch.npu.is_available():
            raise RuntimeError("NPU is unavailable")
        torch.npu.set_device(0)
        result["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                                 "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                                 "triton": triton.__version__,
                                 "triton_ascend_distribution": importlib.metadata.version("triton-ascend"),
                                 "device": torch.npu.get_device_name(0)}
        contract = json.loads((_ROOT / "fixtures/case.json").read_text())
        cases = contract["input_cases"]
        selected = [case for case in cases if case["id"] == "slot_perf_large"] if args.typical else cases
        for case in selected:
            record = _case(case)
            (generalization if case.get("generalization") else result)["cases"].append(record)
        if not args.typical:
            performance = [case for case in cases if case.get("performance")]
            result["negative"] = _negative(performance[0])
            result["dynamic_updates"], result["storage_rebindings"], result["engineering_by_case"] = {}, {}, {}
            for case in performance:
                dynamic, rebindings = _updates(case)
                result["dynamic_updates"][case["id"]] = dynamic
                result["storage_rebindings"][case["id"]] = rebindings
                result["engineering_by_case"][case["id"]] = _engineering(case)
            result["engineering"] = result["engineering_by_case"][performance[0]["id"]]
            result.update(_supplemental())
            tensors, attrs, _ = _inputs(cases[0])
            result["direct_run"] = _check_output(solution.run(tensors, attrs), _oracle(tensors, attrs), tensors)
            _assert_coverage(result, generalization, contract)
        if not args.functional_only:
            for case in selected:
                if case.get("performance") or case.get("generalization", {}).get("measure"):
                    record = {"id": case["id"], "status": "RUNNING"}
                    (result if case.get("performance") else generalization)["benchmarks"].append(record)
                    record.update(_benchmark(case, record))
            if any(record["status"] == "FAIL" for record in result["benchmarks"]):
                raise AssertionError("performance acceptance failed")
        result["status"] = "TYPICAL_PASS" if args.typical else "FUNCTIONAL_ONLY_PASS" if args.functional_only else "PASS"
        exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        result["status"] = "FAIL"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (output / "generalization.json").write_text(json.dumps(generalization, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
