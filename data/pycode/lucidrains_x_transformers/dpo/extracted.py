"""
Extracted Code Preview
============================================================
This file shows all code blocks extracted from source.py.
Review this to verify extraction is correct before running
the synthetic challenge generator.

Total extracted: 1 blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, LayerNorm


# ======================================================================
# EXTRACTED CODE BLOCKS
# ======================================================================

# --- Block 1: DPO (class) ---
# From: x_transformers/dpo.py:51
# Category: other

class DPO(Module):
    def __init__(
        self,
        model: TransformerWrapper,
        *,
        beta = 0.1,
        pad_id = None
    ):
        super().__init__()
        self.policy_model = model

        self.ref_model = deepcopy(model)
        freeze_all_layers_(self.ref_model)

        self.beta = beta
        self.pad_id = pad_id

    def parameters(self):
        return self.policy_model.parameters()

    def forward(
        self,
        preferred_seq,
        unpreferred_seq,
        *,
        prompt_mask,
        preferred_seq_mask = None,
        unpreferred_seq_mask = None,
    ):
        assert preferred_seq.ndim == 2
        assert preferred_seq.shape == unpreferred_seq.shape

        if exists(self.pad_id):
            if not exists(preferred_seq_mask):
                preferred_seq_mask = preferred_seq != self.pad_id

            if not exists(unpreferred_seq_mask):
                unpreferred_seq_mask = unpreferred_seq != self.pad_id

        """
        Following Appendix B in https://arxiv.org/abs/2305.18290
        """

        with torch.no_grad():
            self.ref_model.eval()
            ref_preferred_logprob = log_prob_from_model_and_seq(self.ref_model, preferred_seq)
            ref_unpreferred_logprob = log_prob_from_model_and_seq(self.ref_model, unpreferred_seq)

        policy_preferred_logprob = log_prob_from_model_and_seq(self.policy_model, preferred_seq)
        policy_unpreferred_logprob = log_prob_from_model_and_seq(self.policy_model, unpreferred_seq)

        # masked mean of log probs

        preferred_seq_mask = maybe_and_mask(~prompt_mask, preferred_seq_mask)
        unpreferred_seq_mask = maybe_and_mask(~prompt_mask, unpreferred_seq_mask)

        ref_preferred_logprob, policy_preferred_logprob = map(lambda t: masked_mean(t, preferred_seq_mask), (ref_preferred_logprob, policy_preferred_logprob))
        ref_unpreferred_logprob, policy_unpreferred_logprob = map(lambda t: masked_mean(t, unpreferred_seq_mask), (ref_unpreferred_logprob, policy_unpreferred_logprob))

        # main dpo formula

        policy_logratios = policy_preferred_logprob - policy_unpreferred_logprob
        ref_logratios = ref_preferred_logprob - ref_unpreferred_logprob

        losses = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios))

        return losses.mean()

