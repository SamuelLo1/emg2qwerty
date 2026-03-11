"""Hyperparameter tuning utilities.

This module provides a simple Lightning module wrapper `SimpleConvCTCModule`
that exposes configurable convolutional encoder parameters, and a
small hyperparameter search runner that launches `python -m emg2qwerty.train`
with Hydra overrides for sampled configurations.

Usage (example):
  python -m emg2qwerty.tuning --trials 10 --epochs 10

Notes:
 - The runner invokes the main training script and captures its stdout.
 - Results are saved to `tuning_results.json` in the current working dir.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import shlex
import subprocess
import re
import os
import io
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pathlib import Path
import numpy as np
from emg2qwerty.transforms import Transform

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.lightning import WindowedEMGDataModule
from emg2qwerty.modules import (
	SpectrogramNorm,
	MultiBandRotationInvariantMLP,
)
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from torchmetrics import MetricCollection

from torch.utils.data import Subset
from hydra import initialize, compose, initialize_config_dir
from contextlib import redirect_stdout, redirect_stderr
from omegaconf import OmegaConf
import emg2qwerty.train as train


def _run_and_stream(cmd: List[str], log_path: str) -> tuple[int, str]:
	"""Run `cmd` as subprocess, stream combined stdout/stderr to console,
	write full output to `log_path`, and return (returncode, full_stdout).
	"""
	env = os.environ.copy()
	env.setdefault("PYTHONUNBUFFERED", "1")

	with open(log_path, "w") as lf:
		proc = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
			env=env,
		)
		out_lines: List[str] = []
		if proc.stdout is not None:
			for line in proc.stdout:
				print(line, end="")
				lf.write(line)
				out_lines.append(line)
		proc.wait()
	return proc.returncode, "".join(out_lines)


@dataclass
class Config:
	num_conv_layers: int
	base_filters: int
	kernel_size: int
	pool_size: int  # 0 = no pooling, otherwise kernel size (2 or 4)


class SimpleConvCTCModule(pl.LightningModule):
	"""A small configurable CTC model used for quick tuning runs.

	This class mirrors the training entrypoint's expectations for
	instantiation via Hydra: it accepts optimizer, lr_scheduler and decoder
	configs (which are forwarded by the main script).
	"""

	NUM_BANDS = 2
	ELECTRODE_CHANNELS = 16

	def __init__(
		self,
		in_features: int,
		mlp_features: Sequence[int],
		conv_channels: Sequence[int],
		kernel_size: int = 3,
		pool_size: int = 0,
		optimizer: Any = None,
		lr_scheduler: Any = None,
		decoder: Any = None,
	) -> None:
		super().__init__()
		self.save_hyperparameters()

		num_features = self.NUM_BANDS * mlp_features[-1]

		# Build encoder stack: spectrogram norm -> band MLP -> conv+GRU
		self.model = torch.nn.Sequential(
			SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
			MultiBandRotationInvariantMLP(
				in_features=in_features, mlp_features=mlp_features, num_bands=self.NUM_BANDS
			),
			torch.nn.Flatten(start_dim=2),
			# The conv+recurrent stack will be implemented in forward using
			# a small internal conv/gru so that pooling (optional) can be
			# injected between conv layers.
		)

		# Build conv stack modules to use in forward
		conv_layers: List[torch.nn.Module] = []
		in_ch = num_features
		for out_ch in conv_channels:
			conv_layers.append(torch.nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
			conv_layers.append(torch.nn.ReLU())
			conv_layers.append(torch.nn.BatchNorm1d(out_ch))
			if pool_size and pool_size > 1:
				conv_layers.append(torch.nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
			in_ch = out_ch

		self.conv_net = torch.nn.Sequential(*conv_layers)

		# GRU after conv stack
		self.gru = torch.nn.GRU(input_size=in_ch, hidden_size=256, num_layers=2, bidirectional=True)
		self.dropout = torch.nn.Dropout(0.1)

		self.fc = torch.nn.Linear(256 * 2, charset().num_classes)
		self.log_softmax = torch.nn.LogSoftmax(dim=-1)

		# Criterion and decoder
		self.ctc_loss = torch.nn.CTCLoss(blank=charset().null_class)
		self.decoder = instantiate(decoder) if decoder is not None else None

		# Metrics
		metrics = MetricCollection([CharacterErrorRates()])
		self.metrics = torch.nn.ModuleDict(
			{
				f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
				for phase in ["train", "val", "test"]
			}
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		x = self.model(inputs)
		# x: (T, N, num_features) -> to (N, C_in, T) for conv
		x = x.permute(1, 2, 0)
		x = self.conv_net(x)
		# to (T, N, C_out)
		x = x.permute(2, 0, 1)
		x, _ = self.gru(x)
		x = self.dropout(x)
		x = self.fc(x)
		return self.log_softmax(x)

	def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
		inputs = batch["inputs"]
		targets = batch["targets"]
		input_lengths = batch["input_lengths"]
		target_lengths = batch["target_lengths"]

		emissions = self.forward(inputs)

		T_diff = inputs.shape[0] - emissions.shape[0]
		emission_lengths = input_lengths - T_diff

		loss = self.ctc_loss(
			log_probs=emissions,
			targets=targets.transpose(0, 1),
			input_lengths=emission_lengths,
			target_lengths=target_lengths,
		)


		# Decode emissions (if decoder provided)
		predictions = []
		if self.decoder is not None:
			predictions = self.decoder.decode_batch(
				emissions=emissions.detach().cpu().numpy(),
				emission_lengths=emission_lengths.detach().cpu().numpy(),
			)

		# Update metrics
		metrics = self.metrics[f"{phase}_metrics"]
		if predictions:
			targets_np = targets.detach().cpu().numpy()
			target_lengths_np = target_lengths.detach().cpu().numpy()
			for i in range(len(target_lengths_np)):
				target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
				metrics.update(prediction=predictions[i], target=target)

		self.log(f"{phase}/loss", loss, batch_size=len(input_lengths), sync_dist=False)
		return loss

	def _epoch_end(self, phase: str) -> None:
		metrics = self.metrics[f"{phase}_metrics"]
		self.log_dict(metrics.compute(), sync_dist=False)
		metrics.reset()

	def training_step(self, batch, batch_idx):
		return self._step("train", batch)

	def validation_step(self, batch, batch_idx):
		return self._step("val", batch)

	def test_step(self, batch, batch_idx):
		return self._step("test", batch)

	def on_train_epoch_end(self) -> None:
		self._epoch_end("train")

	def on_validation_epoch_end(self) -> None:
		self._epoch_end("val")

	def on_test_epoch_end(self) -> None:
		self._epoch_end("test")

	def configure_optimizers(self):
		# If hydra supplied optimizer/lr_scheduler configs, prefer the utils helper
		try:
			if self.hparams.optimizer is not None and self.hparams.lr_scheduler is not None:
				return utils.instantiate_optimizer_and_scheduler(self.parameters(), self.hparams.optimizer, self.hparams.lr_scheduler)
		except Exception:
			pass
		return torch.optim.Adam(self.parameters(), lr=1e-3)


class SubsampledWindowedEMGDataModule(WindowedEMGDataModule):
	"""A thin wrapper around `WindowedEMGDataModule` that subsamples the
	training dataset to a given fraction for fast tuning runs.

	Accepts the same constructor args as `WindowedEMGDataModule` plus
	`train_fraction` (float in (0,1]).
	"""

	def __init__(
		self,
		window_length: int,
		padding: tuple[int, int],
		batch_size: int,
		num_workers: int,
		train_sessions: Sequence[Path],
		val_sessions: Sequence[Path],
		test_sessions: Sequence[Path],
		train_transform: Transform[np.ndarray, torch.Tensor],
		val_transform: Transform[np.ndarray, torch.Tensor],
		test_transform: Transform[np.ndarray, torch.Tensor],
		train_fraction: float = 1.0,
	) -> None:
		super().__init__(
			window_length=window_length,
			padding=padding,
			batch_size=batch_size,
			num_workers=num_workers,
			train_sessions=train_sessions,
			val_sessions=val_sessions,
			test_sessions=test_sessions,
			train_transform=train_transform,
			val_transform=val_transform,
			test_transform=test_transform,
		)
		self.train_fraction = float(train_fraction)

	def setup(self, stage: str | None = None) -> None:
		super().setup(stage)

		if 0 < self.train_fraction < 1.0:
			total = len(self.train_dataset)
			k = max(1, int(total * self.train_fraction))
			indices = random.sample(range(total), k=k)
			self.train_dataset = Subset(self.train_dataset, indices)


def _build_override_for_config(cfg: Config, epochs: int, limit_train_batches: float | None = None) -> List[str]:
	conv_channels = [cfg.base_filters * (2**i) for i in range(cfg.num_conv_layers)]
	conv_channels_str = ",".join(str(x) for x in conv_channels)
	overrides = [
		# Use a dedicated Hydra model config that exposes CNN params
		"model=cnn_conv_ctc",
		f"module.in_features=528",
		f"module.mlp_features=[384]",
		f"module.conv_channels=[{conv_channels_str}]",
		f"module.kernel_size={cfg.kernel_size}",
		f"module.pool_size={cfg.pool_size}",
		f"trainer.max_epochs={epochs}",
		# Use the subsampled datamodule config so `datamodule.train_fraction` exists
		"+datamodule=subsampled_windowed",
		f"datamodule.train_fraction={{TRAIN_FRACTION}}",
		"train=True",
		"trainer.accelerator=cpu",
		"trainer.devices=1",
	]
	if limit_train_batches is not None:
		overrides.append(f"trainer.limit_train_batches={limit_train_batches}")
	return overrides


def _run_in_process(overrides: List[str], log_path: str) -> tuple[int, str]:
	"""Compose a Hydra config in-process and call the training `main` directly.

	This avoids spawning a subprocess which reduces overhead for many short
	runs. The function captures stdout/stderr and writes a combined log file.
	Returns (returncode, full_output).
	"""
	# Register resolver used by train.main if not already present
	try:
		OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
	except Exception:
		pass

	config_dir = Path(__file__).resolve().parents[1].joinpath("config")
	full_out = ""
	# Hydra initialize/compose in a context so it's cleaned up after the call
	try:
		with initialize_config_dir(config_dir=str(config_dir), job_name="tuning"):
			cfg = compose(config_name="base", overrides=overrides)
			buf = io.StringIO()
			with open(log_path, "w") as lf, redirect_stdout(buf), redirect_stderr(buf):
				# Call the original, undecorated main function with the composed cfg
				train.main.__wrapped__(cfg)
				full_out = buf.getvalue()
				lf.write(full_out)
			return 0, full_out
	except Exception as exc:
		# capture whatever was written to buffer and append exception
		try:
			full_out = buf.getvalue()
		except Exception:
			full_out = ""
		full_out = full_out + f"\nException during in-process run: {exc}\n"
		with open(log_path, "w") as lf:
			lf.write(full_out)
		return 1, full_out



def run_trials(
	trials: int = 10,
	seed: int = 42,
	train_fraction: float = 1.0,
	initial_epochs: int = 12,
	max_epochs: int = 14,
	in_process: bool = True,
	limit_train_batches: float | None = None,
) -> None:
	"""Run sampling trials with early stopping at 12 epochs and max 14.

	For each sampled configuration we first train for 12 epochs. If the
	validation CER at epoch 12 improves over the best CER seen so far, the
	job is continued/resumed to 14 epochs; otherwise it is discarded.
	"""
	random.seed(seed)

	num_conv_options = [1, 2, 3]
	kernel_options = [3, 5, 7]
	pool_size_options = [0, 2, 4]  # 0 means no pooling
	base_filters_options = [32, 64, 128]

	# Sample a pool of candidate configurations; choose `trials` random
	candidates: List[Config] = []
	for _ in range(max(100, trials * 5)):
		cfg = Config(
			num_conv_layers=random.choice(num_conv_options),
			base_filters=random.choice(base_filters_options),
			kernel_size=random.choice(kernel_options),
			pool_size=random.choice(pool_size_options),
		)
		candidates.append(cfg)

	selected = random.sample(candidates, k=trials)

	results: List[Dict[str, Any]] = []
	best_cer = float("inf")

	def _extract_val_cer(parsed: Any) -> float | None:
		try:
			val = parsed.get("val_metrics")
			if isinstance(val, list) and len(val) > 0:
				# val_metrics is often a list of dicts
				entry = val[0]
			elif isinstance(val, dict):
				entry = val
			else:
				return None
			# Common metric key is 'val/CER' or 'val_metrics/val/CER'
			for key in entry:
				if key.endswith("CER") or "CER" in key:
					return float(entry[key])
		except Exception:
			return None
		return None

	# Use provided epoch values
	# initial_epochs and max_epochs are taken from function args

	for i, cfg in enumerate(selected, 1):
		print(f"Trial {i}/{trials}: {cfg}")

		# 1) initial run for `initial_epochs`
		overrides = _build_override_for_config(cfg, initial_epochs, limit_train_batches)
		# replace placeholder with requested train_fraction
		overrides = [o.replace("{TRAIN_FRACTION}", str(train_fraction)) for o in overrides]
		if in_process:
			print("  running initial run in-process")
			log_path = f"tuning_stdout_trial_{i}_initial.log"
			retcode, full_out = _run_in_process(overrides, log_path)
		else:
			cmd = ["python", "-m", "emg2qwerty.train"] + overrides
			print(f"  initial run cmd: {' '.join(shlex.quote(c) for c in cmd)}")
			# Stream subprocess output live to console and write to a log file
			log_path = f"tuning_stdout_trial_{i}_initial.log"
			retcode, full_out = _run_and_stream(cmd, log_path)

		res_item: Dict[str, Any] = {"config": asdict(cfg), "initial_returncode": retcode}
		res_item["initial_stdout"] = full_out[-10000:]
		res_item["initial_stderr"] = ""

		# Try to parse results from initial run
		parsed = None
		# Try to extract a printed Python dict that contains 'val_metrics'
		try:
			m = re.search(r"\{.*'val_metrics'.*\}", full_out, flags=re.S)
			if m:
				tail = m.group(0)
				parsed = ast.literal_eval(tail)
				res_item["initial_results"] = parsed
			else:
				# fallback: ensure the captured log file exists (already written by _run_and_stream)
				res_item["initial_results_parse_error"] = True
				res_item["initial_stdout_log"] = log_path
		except Exception:
			res_item["initial_results_parse_error"] = True
			res_item["initial_stdout_log"] = log_path

		val_cer = _extract_val_cer(parsed) if parsed is not None else None
		res_item["initial_val_cer"] = val_cer

		# Decide whether to continue to `max_epochs`
		if val_cer is not None and val_cer < best_cer:
			print(f"  Improved CER {val_cer:.4f} < best {best_cer:.4f}; continuing to {max_epochs} epochs")
			best_cer = val_cer

			# Resume / continue training to max_epochs using best checkpoint
			checkpoint = None
			if parsed is not None:
				checkpoint = parsed.get("best_checkpoint")

			overrides2 = _build_override_for_config(cfg, max_epochs, limit_train_batches)
			overrides2 = [o.replace("{TRAIN_FRACTION}", str(train_fraction)) for o in overrides2]
			if checkpoint:
				overrides2.append(f"checkpoint={checkpoint}")
			if in_process:
				print("  continuing run in-process")
				log_path2 = f"tuning_stdout_trial_{i}_final.log"
				retcode2, full_out2 = _run_in_process(overrides2, log_path2)
			else:
				cmd2 = ["python", "-m", "emg2qwerty.train"] + overrides2
				print(f"  continuing run cmd: {' '.join(shlex.quote(c) for c in cmd2)}")
				log_path2 = f"tuning_stdout_trial_{i}_final.log"
				retcode2, full_out2 = _run_and_stream(cmd2, log_path2)

			res_item["final_returncode"] = retcode2
			res_item["final_stdout"] = full_out2[-10000:]
			res_item["final_stderr"] = ""

			# parse final results
			try:
				m2 = re.search(r"\{.*'val_metrics'.*\}", full_out2, flags=re.S)
				if m2:
					parsed2 = ast.literal_eval(m2.group(0))
					res_item["final_results"] = parsed2
					# update best_cer if final is even better
					final_val = _extract_val_cer(parsed2)
					if final_val is not None and final_val < best_cer:
						best_cer = final_val
				else:
					res_item["final_results_parse_error"] = True
					res_item["final_stdout_log"] = log_path2
			except Exception:
				res_item["final_results_parse_error"] = True
				res_item["final_stdout_log"] = log_path2
		else:
			print(f"  Did not improve (val_cer={val_cer}); discarding configuration")
			res_item["discarded"] = True

		results.append(res_item)

		# Save intermediate results
		with open("tuning_results.json", "w") as fh:
			json.dump(results, fh, indent=2)


def _cli() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--trials", type=int, default=10)
	parser.add_argument("--initial-epochs", type=int, default=12, help="Number of epochs for initial evaluation run")
	parser.add_argument("--max-epochs", type=int, default=14, help="Maximum epochs to train if configuration improves")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction of training data to use (0-1]")
	parser.set_defaults(in_process=True)
	parser.add_argument("--no-in-process", dest="in_process", action="store_false", help="Disable in-process training (use subprocess) to match prior behavior")
	parser.add_argument("--limit-train-batches", type=float, default=None, help="Optional trainer.limit_train_batches override (0-1 fractional or int)")
	args = parser.parse_args()
	run_trials(
		trials=args.trials,
		seed=args.seed,
		train_fraction=args.train_fraction,
		initial_epochs=args.initial_epochs,
		max_epochs=args.max_epochs,
		in_process=args.in_process,
		limit_train_batches=args.limit_train_batches,
	)


if __name__ == "__main__":
	_cli()


