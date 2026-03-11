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
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.lightning import WindowedEMGDataModule
from emg2qwerty.modules import (
	SpectrogramNorm,
	MultiBandRotationInvariantMLP,
)


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

		self.log(f"{phase}/loss", loss, batch_size=len(input_lengths), sync_dist=False)
		return loss

	def training_step(self, batch, batch_idx):
		return self._step("train", batch)

	def validation_step(self, batch, batch_idx):
		return self._step("val", batch)

	def test_step(self, batch, batch_idx):
		return self._step("test", batch)

	def configure_optimizers(self):
		# If hydra supplied optimizer/lr_scheduler configs, prefer the utils helper
		try:
			if self.hparams.optimizer is not None and self.hparams.lr_scheduler is not None:
				return utils.instantiate_optimizer_and_scheduler(self.parameters(), self.hparams.optimizer, self.hparams.lr_scheduler)
		except Exception:
			pass
		return torch.optim.Adam(self.parameters(), lr=1e-3)


def _build_override_for_config(cfg: Config, epochs: int) -> List[str]:
	conv_channels = [cfg.base_filters * (2**i) for i in range(cfg.num_conv_layers)]
	conv_channels_str = ",".join(str(x) for x in conv_channels)
	overrides = [
		f"module._target_=emg2qwerty.tuning.SimpleConvCTCModule",
		f"module.in_features=528",
		f"module.mlp_features=[384]",
		f"module.conv_channels=[{conv_channels_str}]",
		f"module.kernel_size={cfg.kernel_size}",
		f"module.pool_size={cfg.pool_size}",
		f"trainer.max_epochs={epochs}",
		"train=True",
		"trainer.accelerator=cpu",
		"trainer.devices=1",
	]
	return overrides


def run_trials(trials: int = 10, epochs: int = 10, seed: int = 42) -> None:
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

	initial_epochs = 12
	max_epochs = 14

	for i, cfg in enumerate(selected, 1):
		print(f"Trial {i}/{trials}: {cfg}")

		# 1) initial run for `initial_epochs`
		overrides = _build_override_for_config(cfg, initial_epochs)
		cmd = ["python", "-m", "emg2qwerty.train"] + overrides
		print(f"  initial run cmd: {' '.join(shlex.quote(c) for c in cmd)}")
		proc = subprocess.run(cmd, capture_output=True, text=True)

		res_item: Dict[str, Any] = {"config": asdict(cfg), "initial_returncode": proc.returncode}
		res_item["initial_stdout"] = proc.stdout[-10000:]
		res_item["initial_stderr"] = proc.stderr[-10000:]

		# Try to parse results from initial run
		parsed = None
		try:
			idx = proc.stdout.rfind("\n{")
			if idx == -1:
				idx = proc.stdout.rfind("{\'val_metrics'")
			if idx != -1:
				tail = proc.stdout[idx:].strip()
				parsed = ast.literal_eval(tail)
				res_item["initial_results"] = parsed
		except Exception:
			res_item["initial_results_parse_error"] = True

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

			overrides2 = _build_override_for_config(cfg, max_epochs)
			if checkpoint:
				overrides2.append(f"checkpoint={checkpoint}")

			cmd2 = ["python", "-m", "emg2qwerty.train"] + overrides2
			print(f"  continuing run cmd: {' '.join(shlex.quote(c) for c in cmd2)}")
			proc2 = subprocess.run(cmd2, capture_output=True, text=True)

			res_item["final_returncode"] = proc2.returncode
			res_item["final_stdout"] = proc2.stdout[-10000:]
			res_item["final_stderr"] = proc2.stderr[-10000:]

			# parse final results
			try:
				idx = proc2.stdout.rfind("\n{")
				if idx == -1:
					idx = proc2.stdout.rfind("{\'val_metrics'")
				if idx != -1:
					tail = proc2.stdout[idx:].strip()
					parsed2 = ast.literal_eval(tail)
					res_item["final_results"] = parsed2
					# update best_cer if final is even better
					final_val = _extract_val_cer(parsed2)
					if final_val is not None and final_val < best_cer:
						best_cer = final_val
			except Exception:
				res_item["final_results_parse_error"] = True
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
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	run_trials(trials=args.trials, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
	_cli()


