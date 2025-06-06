import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)

class BaseReward:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weight = config.get("weight", 1.0)
        
    def calculate(self, *args, **kwargs) -> float:
        raise NotImplementedError("Subclasses must implement the calculate method")

class ComplexityReward(BaseReward):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_complexity = config.get("target_complexity", 0.7)
        self.tolerance = config.get("tolerance", 0.15)
        
    def calculate(self, validation_result: Dict[str, Any]) -> float:
        complexity = validation_result.get("complexity", 0.0)
        reward = math.exp(-((complexity - self.target_complexity) ** 2) / (2 * self.tolerance ** 2))
        return self.weight * reward

class ClarityReward(BaseReward):
    def calculate(self, validation_result: Dict[str, Any]) -> float:
        clarity = validation_result.get("clarity", 0.0)
        return self.weight * clarity

class DiversityReward(BaseReward):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ngram_weights = config.get("ngram_weights", [0.2, 0.4, 0.4])

    def _get_ngrams(self, text: str, n: int) -> set:
        words = text.lower().split()
        if len(words) < n:
            return set()
        return set(zip(*(words[i:] for i in range(n))))

    def _calculate_jaccard_similarity(self, set1: set, set2: set) -> float:
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def calculate(self, task_info: Dict[str, Any], previous_tasks: List[Dict[str, Any]]) -> float:
        task_text = task_info.get("task", "")
        if not previous_tasks:
            return self.weight

        previous_task_texts = [p.get("task", "") for p in previous_tasks]
        max_similarity = 0.0

        for prev_text in previous_task_texts:
            total_similarity = 0.0
            for i, weight in enumerate(self.ngram_weights):
                n = i + 1
                ngrams_current = self._get_ngrams(task_text, n)
                ngrams_prev = self._get_ngrams(prev_text, n)
                sim = self._calculate_jaccard_similarity(ngrams_current, ngrams_prev)
                total_similarity += weight * sim
            
            if total_similarity > max_similarity:
                max_similarity = total_similarity
        
        diversity_score = 1.0 - max_similarity
        return self.weight * diversity_score

class AccuracyReward(BaseReward):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.partial_credit_power = config.get("partial_credit_power", 1.5)

    def calculate(self, validation_result: Dict[str, Any]) -> float:
        if not validation_result.get("is_valid", False):
            return 0.0
        correctness = validation_result.get("correctness", 0.0)
        score = correctness ** self.partial_credit_power
        return self.weight * score

class CoherenceReward(BaseReward):
    def calculate(self, validation_result: Dict[str, Any]) -> float:
        coherence = validation_result.get("coherence", 0.0)
        return self.weight * coherence

class RelevanceReward(BaseReward):
    def calculate(self, validation_result: Dict[str, Any]) -> float:
        relevance = validation_result.get("relevance", 0.0)
        return self.weight * relevance

class StructureReward(BaseReward):
    def calculate(self, validation_result: Dict[str, Any]) -> float:
        structure = validation_result.get("structure", 0.0)
        return self.weight * structure
