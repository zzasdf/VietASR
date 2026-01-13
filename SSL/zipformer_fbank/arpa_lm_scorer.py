"""
ArpaLmScorer - ARPA N-gram LM scorer for beam search
Optimized implementation: hybrid unigram + n-gram scoring
"""

import math
import torch
from typing import List, Tuple, Optional, Dict
import numpy as np


class ArpaLmScorer:
    """
    ARPA N-gram LM scorer compatible with icefall beam search.
    
    Key insight: beam_search uses lm_score[new_token] where new_token is topk from ASR.
    We return full vocab scores (vocab_size,) but only the topk scores matter.
    
    Optimization strategy:
    - Pre-compute unigram scores for base (O(1))
    - For each hypothesis with history, compute n-gram scores only when accessed
    - Cache history → scores mapping to avoid recomputation
    """
    
    def __init__(self, arpa_path: str, sos_id: int, eos_id: int, unk_id: int):
        self.arpa_path = arpa_path
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.lm_scale = 1.0
        self.lm_type = "rnn"  # Compatible with beam_search RNN interface
        
        self.ngrams: Dict[int, Dict] = {}
        self.order = 0
        self.vocab_size = 2000
        
        # Pre-computed scores
        self._unigram_scores: Optional[torch.Tensor] = None
        self._score_cache: Dict[tuple, torch.Tensor] = {}
        
        self._load_arpa()
        self._precompute_unigrams()
        
    def _load_arpa(self):
        """Load ARPA n-gram model."""
        print(f"Loading ARPA from {self.arpa_path}...")
        with open(self.arpa_path, 'r', encoding='utf-8') as f:
            current_order = 0
            count = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith('\\data\\') or line.startswith('\\end\\'):
                    continue
                
                if line.startswith('\\') and line.endswith('-grams:'):
                    current_order = int(line[1:].split('-')[0])
                    self.order = max(self.order, current_order)
                    if current_order not in self.ngrams:
                        self.ngrams[current_order] = {}
                    continue
                
                if 'ngram' in line and '=' in line:
                    continue
                    
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                prob = float(parts[0])
                words = parts[1:1+current_order]
                backoff = 0.0
                if len(parts) > 1 + current_order:
                    backoff = float(parts[-1])
                
                # Parse token IDs
                try:
                    token_ids = tuple(int(w) for w in words)
                except ValueError:
                    # Handle special tokens
                    mapped_ids = []
                    for w in words:
                        if w == "<s>":
                            mapped_ids.append(self.sos_id)
                        elif w == "</s>":
                            mapped_ids.append(self.eos_id)
                        elif w == "<unk>":
                            mapped_ids.append(self.unk_id)
                        else:
                            try:
                                mapped_ids.append(int(w))
                            except:
                                mapped_ids.append(self.unk_id)
                    token_ids = tuple(mapped_ids)
                
                self.ngrams[current_order][token_ids] = (prob, backoff)
                
                # Track max vocab
                for t in token_ids:
                    if t >= self.vocab_size:
                        self.vocab_size = t + 1
                count += 1

        total = sum(len(v) for v in self.ngrams.values())
        print(f"Loaded {self.order}-gram LM with {total:,} n-grams, vocab_size={self.vocab_size}")

    def _precompute_unigrams(self):
        """Pre-compute unigram log probs (in natural log) for all vocab tokens."""
        # Use log10 * ln(10) to convert to natural log
        ln10 = 2.302585093
        self._unigram_scores = torch.full((self.vocab_size,), -99.0 * ln10, dtype=torch.float32)
        
        if 1 in self.ngrams:
            for (token_id,), (prob_log10, _) in self.ngrams[1].items():
                if 0 <= token_id < self.vocab_size:
                    self._unigram_scores[token_id] = prob_log10 * ln10
        
        print(f"Pre-computed unigram scores for {self.vocab_size} tokens")

    def _get_prob_log10(self, history: Tuple[int, ...], token: int) -> float:
        """
        Get log10 probability P(token|history) with backoff.
        Returns log10 probability.
        """
        ngram = history + (token,)
        n = len(ngram)
        
        # Check if exact n-gram exists
        if n in self.ngrams and ngram in self.ngrams[n]:
            return self.ngrams[n][ngram][0]
        
        # Backoff
        if len(history) > 0:
            backoff_wt = 0.0
            if len(history) in self.ngrams and history in self.ngrams[len(history)]:
                backoff_wt = self.ngrams[len(history)][history][1]
            return backoff_wt + self._get_prob_log10(history[1:], token)
        
        # Unigram fallback
        if 1 in self.ngrams and (token,) in self.ngrams[1]:
            return self.ngrams[1][(token,)][0]
        
        # OOV
        return -99.0

    def _compute_all_scores(self, history: Tuple[int, ...], vocab_size: int) -> torch.Tensor:
        """
        Compute scores for all vocab tokens given history.
        Returns tensor of shape (vocab_size,) with ln probabilities.
        
        OPTIMIZATION: Cache results for reuse.
        """
        # Check cache first
        cache_key = (history, vocab_size)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        
        ln10 = 2.302585093
        
        if len(history) == 0:
            # No history - return unigrams
            scores = self._unigram_scores[:vocab_size].clone()
        else:
            # Compute n-gram scores for each token
            scores = torch.zeros(vocab_size, dtype=torch.float32)
            for token_id in range(vocab_size):
                prob_log10 = self._get_prob_log10(history, token_id)
                scores[token_id] = prob_log10 * ln10
        
        # Cache (limit size to avoid memory issues)
        if len(self._score_cache) < 50000:
            self._score_cache[cache_key] = scores
        
        return scores

    def score_token(
        self, 
        token_list: torch.Tensor, 
        x_lens: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        vocab_size: int = 2000
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Score tokens and return next token probabilities.
        
        Interface compatible with icefall RnnLmModel.score_token():
        - Input: token_list (batch, 1) - single tokens per hypothesis
        - Output: scores (batch, vocab_size) - log probs for all next tokens
        - State: tuple of (h, c) tensors for RNN-like interface
        
        The state encodes the token history for n-gram lookup.
        h tensor stores the history tokens (last order-1 tokens).
        """
        batch_size = token_list.size(0)
        device = token_list.device
        vocab_size = min(vocab_size, self.vocab_size)
        max_hist_len = self.order - 1
        
        # Extract new tokens
        new_tokens = token_list[:, 0].cpu().tolist()
        
        # Build/update history from state
        histories = []
        if state is None:
            # First call - no history
            for _ in range(batch_size):
                histories.append(())
        else:
            # Extract history from state tensor
            h_state = state[0]  # shape: (1, batch_size, max_hist_len)
            for b in range(batch_size):
                hist_tensor = h_state[0, b, :].cpu().tolist()
                # Filter out padding zeros and convert to ints
                hist = tuple(int(t) for t in hist_tensor if t > 0)
                histories.append(hist)
        
        # Compute scores for each hypothesis
        all_scores = []
        new_histories = []
        
        for b in range(batch_size):
            history = histories[b]
            new_token = new_tokens[b]
            
            # Update history with new token
            new_hist = history + (new_token,)
            if len(new_hist) > max_hist_len:
                new_hist = new_hist[-max_hist_len:]
            new_histories.append(new_hist)
            
            # Get scores for all next tokens given new history
            scores = self._compute_all_scores(new_hist, vocab_size)
            all_scores.append(scores)
        
        # Stack scores
        scores_tensor = torch.stack(all_scores, dim=0).to(device)  # (batch, vocab_size)
        
        # Encode new histories into state tensors
        new_h = torch.zeros((1, batch_size, max_hist_len), dtype=torch.float32, device=device)
        new_c = torch.zeros_like(new_h)  # Not used but needed for interface
        
        for b, hist in enumerate(new_histories):
            for i, t in enumerate(hist):
                if i < max_hist_len:
                    new_h[0, b, i] = float(t)
        
        return scores_tensor, (new_h, new_c)
