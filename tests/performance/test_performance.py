import os
import time
import json
import argparse
from pathlib import Path
from typing import Any

from collections.abc import Generator
from collections import defaultdict
import psutil
import numpy as np
from contextlib import contextmanager
from tabulate import tabulate

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPU_AVAILABLE = False

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import get_settings
from culicidaelab.predictors import MosquitoClassifier, MosquitoDetector, MosquitoSegmenter


@contextmanager
def temp_env_vars(new_vars: dict[str, str]) -> Generator[None, Any, None]:
    """
    A context manager to temporarily set environment variables and
    restore the original state upon exit.
    """
    original_vars = {key: os.getenv(key) for key in new_vars}

    print("---")
    print("Temporarily setting environment variables for the test:")

    for key, value in new_vars.items():
        print(f"  - Setting {key}={value} (Original was: {original_vars[key]})")
        os.environ[key] = str(value)
    print("---")

    try:
        yield
    finally:
        print("---")
        print("Restoring original environment variables...")
        for key, original_value in original_vars.items():
            if original_value is None:  #
                print(f"  - Unsetting {key}")
                os.environ.pop(key, None)
            else:
                print(f"  - Restoring {key}={original_value}")
                os.environ[key] = original_value
        print("Test environment cleaned up.")
        print("---")


@contextmanager
def measure_performance() -> Generator[dict[str, Any], None, None]:
    metrics = {}
    process = psutil.Process(os.getpid())
    gpus = GPUtil.getGPUs() if GPU_AVAILABLE else []

    cpu_times_start = process.cpu_times()
    mem_rss_start = process.memory_info().rss
    gpu_mem_start = {gpu.id: gpu.memoryUsed for gpu in gpus} if gpus else {}

    start_time = time.perf_counter()
    try:
        yield metrics
    finally:
        end_time = time.perf_counter()
        mem_rss_end = process.memory_info().rss
        cpu_times_end = process.cpu_times()
        gpus_after = GPUtil.getGPUs() if GPU_AVAILABLE else []
        gpu_mem_end = {gpu.id: gpu.memoryUsed for gpu in gpus_after} if gpus_after else {}
        gpu_util_end = {gpu.id: gpu.load for gpu in gpus_after} if gpus_after else {}

        metrics["duration_sec"] = end_time - start_time
        metrics["cpu_time_sec"] = (cpu_times_end.user - cpu_times_start.user) + (
            cpu_times_end.system - cpu_times_start.system
        )
        metrics["mem_rss_bytes_diff"] = mem_rss_end - mem_rss_start
        metrics["mem_rss_bytes_final"] = mem_rss_end

        if gpus:
            metrics["gpu_memory_mb_diff"] = {
                gpu_id: gpu_mem_end.get(gpu_id, 0) - gpu_mem_start.get(gpu_id, 0) for gpu_id in gpu_mem_start
            }
            metrics["gpu_utilization_percent"] = gpu_util_end
            metrics["gpu_memory_mb_final"] = gpu_mem_end
        else:
            metrics["gpu_memory_mb_diff"] = {}
            metrics["gpu_utilization_percent"] = {}
            metrics["gpu_memory_mb_final"] = {}


class PerformanceTester:
    def __init__(
        self,
        model_name: str,
        device: str,
        image_size: int,
        num_runs: int,
        warmup_runs: int,
        output_dir: Path,
    ):
        self.model_name = model_name
        self.device = device
        self.image_size = image_size
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.output_path = output_dir / f"{model_name}_performance.json"
        self.results: list[dict] = []
        self.settings = get_settings()

    def _get_model(self, load: bool = False) -> BasePredictor:
        model_map = {
            "classifier": MosquitoClassifier,
            "detector": MosquitoDetector,
            "segmenter": MosquitoSegmenter,
        }
        if self.model_name not in model_map:
            raise ValueError(f"Model '{self.model_name}' not recognized.")

        config_path = f"predictors.{self.model_name}.device"
        print(f"Forcing device setting: '{config_path}' = '{self.device}'")
        self.settings.set_config(config_path, self.device)

        return model_map[self.model_name](settings=self.settings, load_model=load)

    def _generate_mock_data(self, batch_size: int) -> list[np.ndarray]:
        shape = (self.image_size, self.image_size, 3)
        return [np.random.randint(0, 255, shape, dtype=np.uint8) for _ in range(batch_size)]

    def _run_and_average(self, test_name: str, func: Any, *args, **kwargs) -> dict:
        print(f"\nRunning test: '{test_name}'...")

        for i in range(self.warmup_runs):
            print(f"  Warm-up run {i+1}/{self.warmup_runs}...")
            _ = func(*args, **kwargs)

        collected_metrics = defaultdict(list)
        last_run_metrics = {}
        for i in range(self.num_runs):
            print(f"  Measurement run {i+1}/{self.num_runs}...")
            with measure_performance() as metrics:
                func(*args, **kwargs)
            last_run_metrics = metrics

            for key, value in metrics.items():
                if not isinstance(value, dict):
                    collected_metrics[key].append(value)

        averaged_metrics = {key: np.mean(val) for key, val in collected_metrics.items()}
        averaged_metrics.update({k: v for k, v in last_run_metrics.items() if isinstance(v, (dict, int))})

        result_log = {"test_name": test_name, **kwargs, **averaged_metrics}
        self.results.append(result_log)
        return result_log

    def test_model_loading(self):
        self._run_and_average("model_loading", self._get_model, load=True)

    def test_prediction(self, batch_sizes: list[int]):
        model = self._get_model(load=True)

        for bs in batch_sizes:
            input_data = self._generate_mock_data(bs)
            test_name = f"prediction_batch_{bs}"

            if bs == 1:
                self._run_and_average(test_name, model.predict, input_data[0], batch_size=bs)
            else:
                self._run_and_average(test_name, model.predict_batch, input_data, batch_size=bs)

    def save_results(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nAll results saved to {self.output_path}")

    def print_summary(self):
        if not self.results:
            print("No results to summarize.")
            return

        headers = [
            "Test Name",
            "Batch Size",
            "Avg Time (s)",
            "Avg CPU Time (s)",
            "RAM Final (MB)",
            "GPU Mem Final (MB)",
        ]
        rows = []
        for res in self.results:
            batch_size = res.get("batch_size", "N/A")
            gpu_mem = res.get("gpu_memory_mb_final", {})
            gpu_mem_str = str(list(gpu_mem.values())[0]) if gpu_mem else "N/A"

            rows.append(
                [
                    res["test_name"],
                    batch_size,
                    f"{res.get('duration_sec', 0):.4f}",
                    f"{res.get('cpu_time_sec', 0):.4f}",
                    f"{res.get('mem_rss_bytes_final', 0) / 1e6:.2f}",
                    gpu_mem_str,
                ],
            )

        print("\n--- Performance Summary ---")
        print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(description="CulicidaeLab Performance Testing Suite")
    parser.add_argument("--model-name", type=str, required=True, choices=["classifier", "detector", "segmenter"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/performance/performance_logs"))

    parser.add_argument(
        "--limit-threads",
        action="store_true",
        help="Limit MKL/OMP threads to 1 to reduce memory usage and get stable CPU results.",
    )

    args = parser.parse_args()

    limited_env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_SERVICE_FORCE_INTEL": "1",
    }

    def run_tests():
        if args.device == "cuda" and not GPU_AVAILABLE:
            print("Warning: --device cuda was requested, but no GPU or GPUtil library was found. Forcing CPU.")
            args.device = "cpu"

        tester = PerformanceTester(
            model_name=args.model_name,
            device=args.device,
            image_size=args.image_size,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            output_dir=args.output_dir,
        )

        tester.test_model_loading()
        tester.test_prediction(batch_sizes=args.batch_sizes)
        tester.print_summary()
        tester.save_results()

    if args.limit_threads:
        with temp_env_vars(limited_env_vars):
            run_tests()
    else:
        print("---")
        print("Running test with default system thread settings.")
        print("---")
        run_tests()


if __name__ == "__main__":
    main()
