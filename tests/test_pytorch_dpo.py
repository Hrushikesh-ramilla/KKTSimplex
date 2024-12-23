"""
Unit tests for PyTorch DPO Loss and Synthetic Dataset Generation.
"""

import os
import json
import tempfile
import pytest
import torch
from kkt_simplex.dpo_dataset import (
    DPOAntiSycophancyLoss,
    PreferencePair,
    generate_anti_sycophancy_pairs,
    export_dpo_dataset_jsonl,
)


def test_pytorch_dpo_loss_forward_and_backward():
    """Verify DPOAntiSycophancyLoss calculates correct scalar loss and backward gradients."""
    dpo_loss = DPOAntiSycophancyLoss(beta=0.1, label_smoothing=0.0)

    # Batch of 4 samples
    batch_size = 4
    policy_chosen_logps = torch.tensor([-2.1, -1.8, -3.0, -2.5], requires_grad=True)
    policy_rejected_logps = torch.tensor([-5.2, -4.9, -6.1, -5.8], requires_grad=True)
    reference_chosen_logps = torch.tensor([-2.2, -1.9, -3.1, -2.6])
    reference_rejected_logps = torch.tensor([-5.0, -4.8, -6.0, -5.7])

    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0

    # Chosen rewards must be higher than rejected rewards
    assert (chosen_rewards > rejected_rewards).all()

    # Verify backpropagation
    loss.backward()
    assert policy_chosen_logps.grad is not None
    assert policy_rejected_logps.grad is not None
    # Chosen gradient is negative (loss decreases as chosen logp increases)
    assert (policy_chosen_logps.grad < 0).all()
    # Rejected gradient is positive (loss increases as rejected logp increases)
    assert (policy_rejected_logps.grad > 0).all()


def test_pytorch_dpo_label_smoothing():
    """Verify label smoothing regularizes loss output."""
    dpo_loss_clean = DPOAntiSycophancyLoss(beta=0.1, label_smoothing=0.0)
    dpo_loss_smoothed = DPOAntiSycophancyLoss(beta=0.1, label_smoothing=0.1)

    p_c = torch.tensor([-1.0])
    p_r = torch.tensor([-5.0])
    r_c = torch.tensor([-2.0])
    r_r = torch.tensor([-4.0])

    loss_clean, _, _ = dpo_loss_clean(p_c, p_r, r_c, r_r)
    loss_smoothed, _, _ = dpo_loss_smoothed(p_c, p_r, r_c, r_r)

    assert loss_smoothed.item() > loss_clean.item()


def test_generate_and_export_dpo_dataset():
    """Verify export_dpo_dataset_jsonl produces valid JSONL entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_dpo.jsonl")
        num_samples = 25

        export_dpo_dataset_jsonl(out_path, num_samples=num_samples)
        assert os.path.isfile(out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == num_samples
        first = lines[0]
        assert "prompt" in first
        assert "chosen" in first
        assert "rejected" in first
        assert "metadata" in first
        assert first["metadata"]["ground_truth_intact"] is True
        assert first["metadata"]["sycophantic_flaw"] is not None
