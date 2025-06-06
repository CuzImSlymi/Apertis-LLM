import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import re
import string
import math

logger = logging.getLogger(__name__)

STOP_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot',
    'com', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'else', 'ever', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't",
    'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his',
    'how', "how's", 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
    'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some',
    'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's",
    'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what',
    "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",
    'with', "won't", 'would', "wouldn't", 'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
    'yourself', 'yourselves'
])

class TaskGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_types = config.get("task_types", ["abduction", "deduction", "induction"])
        self.task_distribution = config.get("task_distribution", [0.3, 0.3, 0.4])
        self.max_attempts = config.get("max_attempts", 3)
        self.seed_tasks = self._load_seed_tasks(config.get("seed_tasks_path"))
        
    def _load_seed_tasks(self, path: Optional[str]) -> Dict[str, List[str]]:
        if not path or not os.path.exists(path):
            logger.warning(f"Seed tasks path not provided or does not exist: {path}")
            return {task_type: [] for task_type in self.task_types}
        try:
            tasks = {task_type: [] for task_type in self.task_types}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        task_type = item.get("type", "")
                        if task_type in self.task_types and "task" in item:
                            tasks[task_type].append(item["task"])
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in seed tasks file: {line}")
            return tasks
        except Exception as e:
            logger.error(f"Error loading seed tasks: {e}")
            return {task_type: [] for task_type in self.task_types}
    
    def generate_task(self, model, hf_tokenizer) -> Dict[str, Any]:
        import numpy as np
        task_type = np.random.choice(self.task_types, p=self.task_distribution)
        
        if self.seed_tasks.get(task_type) and np.random.random() < self.config.get("seed_task_probability", 0.2):
            seed_tasks_for_type = self.seed_tasks.get(task_type, [])
            if seed_tasks_for_type:
                task = np.random.choice(seed_tasks_for_type)
                return {"task": task, "type": task_type, "from_seed": True}
        
        prompt = self._get_task_generation_prompt(task_type)
        device = self._get_model_device(model)

        for attempt in range(self.max_attempts):
            try:
                inputs = hf_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=model.config.max_position_embeddings // 2
                ) 
                
                input_ids_for_generation = inputs["input_ids"].to(device)
                attention_mask_for_generation = inputs.get("attention_mask", torch.ones_like(input_ids_for_generation)).to(device)

                outputs_tensor = model.generate(
                    input_ids=input_ids_for_generation,
                    attention_mask=attention_mask_for_generation,
                    max_new_tokens=self.config.get("max_new_tokens", 512),
                    temperature=self.config.get("temperature", 0.7),
                    top_p=self.config.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 0,
                    eos_token_id=hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else model.config.eos_token_id
                )
                
                generated_ids = outputs_tensor[0, input_ids_for_generation.shape[1]:]
                generated_text_str = hf_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                task = self._extract_task(generated_text_str)
                if task:
                    if len(task) < 15:
                        task = f"Create a more detailed and challenging {task_type} reasoning problem based on the concept of: '{task}'."
                    return {"task": task, "type": task_type, "from_seed": False}
                    
                logger.warning(f"Failed to extract task from generated text (attempt {attempt+1}/{self.max_attempts})")
            except Exception as e:
                logger.error(f"Error generating task (attempt {attempt+1}/{self.max_attempts}): {e}", exc_info=True)
                
        seed_tasks_for_type_fallback = self.seed_tasks.get(task_type, [])
        if seed_tasks_for_type_fallback:
            task = np.random.choice(seed_tasks_for_type_fallback)
            logger.warning(f"Using seed task after {self.max_attempts} failed generation attempts")
            return {"task": task, "type": task_type, "from_seed": True}
            
        logger.info(f"All task generation attempts failed, using placeholder task")
        return {
            "task": f"Create a simple {task_type} reasoning problem about numbers. For example, if we have a sequence 2, 4, 6, 8, what comes next and why?",
            "type": task_type,
            "from_seed": False,
            "is_placeholder": True
        }
    
    def _get_model_device(self, model) -> torch.device:
        if hasattr(model, "device"): return model.device
        if hasattr(model, "parameters"):
            try: return next(model.parameters()).device
            except StopIteration: pass
        return torch.device("cpu")
    
    def _get_task_generation_prompt(self, task_type: str) -> str:
        base_prompt = self.config.get("base_prompt", "You are a problem designer. Generate a novel, challenging reasoning problem that requires deep thinking.")
        prompts = {
            "abduction": " The problem must require abductive reasoning, where one infers the most plausible explanation from a set of observations. The answer should not be immediately obvious. Example: 'A detective finds a room with a shattered window, a valuable painting missing, and a single muddy footprint near the window. What is the most likely sequence of events?'",
            "deduction": " The problem must require deductive reasoning, where a conclusion is logically derived from a set of premises. It should involve multiple steps of reasoning. Example: 'All expert systems are intelligent. All intelligent systems use knowledge. Apertis is an expert system. What can you deduce about Apertis?'",
            "induction": " The problem must require inductive reasoning, where a general rule is inferred from specific examples. The pattern should be non-trivial. Example: 'Consider the sequence: 3, 7, 16, 35, 74. What is the next number and what is the rule governing the sequence?'"
        }
        return base_prompt + prompts.get(task_type, "")

    def _extract_task(self, text: str) -> Optional[str]:
        lines = text.strip().split('\n')
        if not lines: return None
        task = lines[0]
        for prefix in ["Task:", "Problem:", "Question:"]:
            if task.startswith(prefix):
                task = task[len(prefix):].strip()
        if len(task) < 15 and len(lines) > 1:
            task = " ".join(lines[:min(3, len(lines))])      
        return task if task else None

class TaskValidator:
    def __init__(self, config: Dict[str, Any], python_executor=None):
        self.config = config
        self.python_executor = python_executor
        self.min_length = config.get("min_length", 15)
        self.max_length = config.get("max_length", 2500)
        self.logical_operators = {'and', 'or', 'not', 'if', 'then', 'all', 'some', 'none', 'every', 'any'}
        self.comparative_operators = {'<', '>', '<=', '>=', '==', '!='}
        
    def validate(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        task = task_info.get("task", "")
        task_type = task_info.get("type", "")
        if task_info.get("is_placeholder", False):
            return {"is_valid": True, "complexity": 0.5, "clarity": 0.5, "executability": 1.0, "reason": "Placeholder task accepted"}
        if not task or len(task) < self.min_length: return {"is_valid": False, "reason": "Task too short"}
        if len(task) > self.max_length: return {"is_valid": False, "reason": "Task too long"}
        
        executability = {"is_executable": True, "score": 1.0, "reason": None}
        if task_type == "induction" and self.python_executor:
            executability = self._check_executability(task)

        complexity = self._calculate_complexity(task)
        clarity = self._calculate_clarity(task)

        is_valid = (
            complexity >= self.config.get("min_complexity", 0.2) and
            clarity >= self.config.get("min_clarity", 0.4)
        )
        if not is_valid and task_info.get("from_seed", False):
            is_valid = True
            logger.info("Accepting seed task despite not meeting thresholds")
            
        logger.info(f"Task validation metrics - complexity: {complexity:.3f}, clarity: {clarity:.3f}, valid: {is_valid}")
        return {"is_valid": is_valid, "complexity": complexity, "clarity": clarity, 
                "executability": executability["score"],
                "reason": None if is_valid else "Failed metric thresholds"}
    
    def _calculate_complexity(self, task: str) -> float:
        words = task.lower().split()
        word_count = len(words)
        if word_count == 0: return 0.0

        unique_words = len(set(words))
        long_words = sum(1 for word in words if len(word) > 8)
        numbers = len(re.findall(r'\b\d+\.?\d*\b', task))
        logical_ops_count = sum(1 for word in words if word in self.logical_operators)
        math_ops_count = sum(task.count(op) for op in ['+', '-', '*', '/', '^', '='])
        comparative_ops_count = sum(task.count(op) for op in self.comparative_operators)
        code_blocks = len(self._extract_code_blocks(task))
        
        word_count_score = min(1.0, word_count / 150)
        lexical_density_score = min(1.0, (unique_words / word_count if word_count > 0 else 0) * 1.5)
        long_word_score = min(1.0, long_words / 10)
        numerical_score = min(1.0, numbers / 8)
        reasoning_ops_score = min(1.0, (logical_ops_count + comparative_ops_count) / 10)
        math_score = min(1.0, math_ops_count / 8)
        code_score = min(1.0, code_blocks * 0.5)

        complexity = (
            word_count_score * 0.15 +
            lexical_density_score * 0.20 +
            long_word_score * 0.15 +
            numerical_score * 0.15 +
            reasoning_ops_score * 0.20 +
            math_score * 0.10 +
            code_score * 0.05
        )
        return max(0.1, complexity)
    
    def _syllable_count(self, word: str) -> int:
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word and word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1
        return max(1, count)

    def _calculate_clarity(self, task: str) -> float:
        sentences = [s for s in re.split(r'[.!?]+', task) if len(s.strip()) > 3]
        sentence_count = len(sentences)
        words = [w.strip(string.punctuation) for w in task.split() if w.strip(string.punctuation)]
        word_count = len(words)

        if word_count < 5 or sentence_count < 1: return 0.0

        avg_sentence_length = word_count / sentence_count
        
        syllables = sum(self._syllable_count(word) for word in words)
        avg_syllables_per_word = syllables / word_count if word_count > 0 else 0
        
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        flesch_clarity = max(0.0, min(1.0, flesch_score / 100.0))

        question_score = 1.0 if '?' in task else 0.7
        
        clarity = (flesch_clarity * 0.85) + (question_score * 0.15)
        return clarity
    
    def _check_executability(self, task: str) -> Dict[str, Any]:
        if not self.python_executor: return {"is_executable": True, "score": 1.0, "reason": "No executor"}
        code_blocks = self._extract_code_blocks(task)
        if not code_blocks: return {"is_executable": True, "score": 0.5, "reason": "No code found"}
        
        for code in code_blocks:
            result = self.python_executor.execute(code)
            if not result["success"]:
                return {"is_executable": False, "score": 0.0, "reason": f"Execution failed: {result.get('error', 'Unknown')}"}
        return {"is_executable": True, "score": 1.0, "reason": "Execution successful"}
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

class SolutionGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_attempts = config.get("max_attempts", 3)
        
    def generate_solution(self, task_info: Dict[str, Any], model, hf_tokenizer) -> Dict[str, Any]:
        task = task_info.get("task", "")
        task_type = task_info.get("type", "")
        prompt = self._get_solution_generation_prompt(task, task_type)
        device = self._get_model_device(model)

        for attempt in range(self.max_attempts):
            try:
                inputs = hf_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=model.config.max_position_embeddings // 2
                )
                
                input_ids_for_generation = inputs["input_ids"].to(device)
                attention_mask_for_generation = inputs.get("attention_mask", torch.ones_like(input_ids_for_generation)).to(device)
                
                outputs_tensor = model.generate(
                    input_ids=input_ids_for_generation,
                    attention_mask=attention_mask_for_generation,
                    max_new_tokens=self.config.get("max_new_tokens", 1024),
                    temperature=self.config.get("temperature", 0.6),
                    top_p=self.config.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 0,
                    eos_token_id=hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else model.config.eos_token_id
                )
                
                generated_ids = outputs_tensor[0, input_ids_for_generation.shape[1]:]
                generated_text_str = hf_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                solution = self._extract_solution(generated_text_str)
                if solution:
                    return {"task": task, "type": task_type, "solution": solution, "raw_generation": generated_text_str}
                    
                logger.warning(f"Failed to extract solution from generated text (attempt {attempt+1}/{self.max_attempts})")
            except Exception as e:
                logger.error(f"Error generating solution (attempt {attempt+1}/{self.max_attempts}): {e}", exc_info=True)
                
        logger.info(f"All solution generation attempts failed, using placeholder solution")
        placeholders = {
            "abduction": f"To solve this abductive reasoning problem, the most plausible explanation must be inferred from the key observations in '{task[:60]}...'. This involves forming and evaluating hypotheses based on the provided evidence.",
            "deduction": f"To solve this deductive reasoning problem, logical rules must be strictly applied to the premises given in '{task[:60]}...'. The conclusion must necessarily follow if the premises are true.",
            "induction": f"To solve this inductive reasoning problem, a general pattern must be identified from the specific instances mentioned in '{task[:60]}...'. This pattern can then be used to predict future outcomes or formulate a general rule.",
            "default": "To solve this problem, I will first break it down into its core components, analyze the relationships between them, and then apply a step-by-step reasoning process to arrive at a logical conclusion."
        }
        placeholder = placeholders.get(task_type, placeholders["default"])
        return {"task": task, "type": task_type, "solution": placeholder, "raw_generation": "", "is_placeholder": True}
    
    def _get_model_device(self, model) -> torch.device:
        if hasattr(model, "device"): return model.device
        if hasattr(model, "parameters"):
            try: return next(model.parameters()).device
            except StopIteration: pass
        return torch.device("cpu")
    
    def _get_solution_generation_prompt(self, task: str, task_type: str) -> str:
        base_prompt = self.config.get("base_prompt", "You are a world-class reasoning expert. Solve the following problem by thinking step-by-step. Provide a clear, structured, and detailed explanation. Enclose your final answer within <answer> tags.")
        if self.config.get("include_task_type_hint", True):
            type_hint = f" This is a {task_type} reasoning problem."
            base_prompt += type_hint
        return f"{base_prompt}\n\n### Problem ###\n{task}\n\n### Solution ###"
    
    def _extract_solution(self, text: str) -> Optional[str]:
        if not text.strip(): return None
        solution_pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(solution_pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

class SolutionValidator:
    def __init__(self, config: Dict[str, Any], python_executor=None):
        self.config = config
        self.python_executor = python_executor
        
    def validate(self, task_info: Dict[str, Any], solution_info: Dict[str, Any]) -> Dict[str, Any]:
        task = task_info.get("task", "")
        task_type = task_info.get("type", "")
        solution = solution_info.get("solution", "")
        raw_generation = solution_info.get("raw_generation", solution)

        if solution_info.get("is_placeholder", False):
            return {"is_valid": True, "correctness": 0.5, "coherence": 0.5, "relevance": 0.5, "structure": 0.5, "reason": "Placeholder solution accepted"}
        if not solution: return {"is_valid": False, "correctness": 0.0, "reason": "Empty solution"}
        
        if task_type == "induction" and self.python_executor:
            return self._validate_with_execution(task, solution, raw_generation)
        return self._validate_with_heuristics(task, task_type, solution, raw_generation)
    
    def _validate_with_execution(self, task: str, solution: str, raw_generation: str) -> Dict[str, Any]:
        if not self.python_executor: 
            return self._validate_with_heuristics(task, "induction", solution, raw_generation)
        code_blocks = self._extract_code_blocks(solution)
        if not code_blocks:
            logger.warning("No code found in induction solution for execution, using heuristic validation.")
            return self._validate_with_heuristics(task, "induction", solution, raw_generation)
        
        execution_success = True
        for code in code_blocks:
            result = self.python_executor.execute(code)
            if not result["success"]:
                logger.warning(f"Code execution failed but continuing heuristic validation: {result.get('error', 'Unknown error')}")
                execution_success = False
                break
        
        metrics = self._validate_with_heuristics(task, "induction", solution, raw_generation, is_executed=True)
        if execution_success:
            metrics["correctness"] = max(metrics["correctness"], 0.8) 
            metrics["reason"] = "Execution successful, combined with heuristics"
        else:
            metrics["correctness"] *= 0.5 
            metrics["reason"] = "Execution failed, heuristic score penalized"
        
        metrics["is_valid"] = metrics["correctness"] >= self.config.get("min_correctness_exec", 0.4)
        return metrics
    
    def _validate_with_heuristics(self, task: str, task_type: str, solution: str, raw_generation: str, is_executed: bool = False) -> Dict[str, Any]:
        words = solution.split()
        word_count = len(words)
        if word_count < 10: return {"is_valid": False, "correctness": 0.0, "reason": "Solution too short"}
        
        coherence = self._calculate_coherence(solution)
        relevance = self._calculate_relevance(task, solution)
        structure = self._calculate_structure(raw_generation)
        
        correctness = (coherence + relevance + structure) / 3
        
        is_valid = (
            coherence >= self.config.get("min_coherence", 0.4) and
            relevance >= self.config.get("min_relevance", 0.4) and
            structure >= self.config.get("min_structure", 0.3) and
            correctness >= self.config.get("min_correctness", 0.45)
        )

        if not is_valid and word_count >= 50 and not is_executed:
            is_valid = True
            logger.info("Accepting lengthy solution despite not meeting thresholds")

        logger.info(f"Solution validation - coherence: {coherence:.3f}, relevance: {relevance:.3f}, structure: {structure:.3f}, valid: {is_valid}")
        return {"is_valid": is_valid, "correctness": correctness, "coherence": coherence, 
                "relevance": relevance, "structure": structure, "reason": None if is_valid else "Failed heuristic metric thresholds"}
    
    def _calculate_coherence(self, solution: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', solution) if len(s.strip().split()) > 3]
        sentence_count = len(sentences)
        if sentence_count <= 1: return 0.3
        
        transition_words = [
            "therefore", "thus", "hence", "consequently", "as a result", "because",
            "first", "second", "third", "finally", "in conclusion",
            "however", "although", "conversely", "similarly", "likewise", "in addition", "furthermore"
        ]
        transition_count = sum(solution.lower().count(word) for word in transition_words)
        transition_factor = min(1.0, transition_count / (sentence_count * 0.25))

        word_sets = [set(s.lower().split()) - STOP_WORDS for s in sentences]
        overlap_scores = []
        for i in range(sentence_count - 1):
            set1, set2 = word_sets[i], word_sets[i+1]
            if not set1 or not set2: continue
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            overlap_scores.append(intersection / union if union > 0 else 0)
        
        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
        
        return 0.2 + (transition_factor * 0.4) + (avg_overlap * 0.6)

    def _calculate_relevance(self, task: str, solution: str) -> float:
        task_words = set(w.lower() for w in task.split() if w.lower() not in STOP_WORDS)
        solution_words = set(w.lower() for w in solution.split() if w.lower() not in STOP_WORDS)
        if not task_words: return 0.5
        
        overlap = task_words.intersection(solution_words)
        
        if not task_words: return 0.0
        jaccard_similarity = len(overlap) / len(task_words.union(solution_words))
        
        return min(1.0, 0.2 + jaccard_similarity * 2.0)

    def _calculate_structure(self, solution: str) -> float:
        lines = solution.split('\n')
        line_count = len(lines)
        if line_count <= 1: return 0.2
        
        non_empty_lines = [line for line in lines if line.strip()]
        
        numbered_lines = sum(1 for line in non_empty_lines if re.match(r'^\s*\d+[.)]', line))
        bullet_points = sum(1 for line in non_empty_lines if re.match(r'^\s*[-*•]', line))
        conclusion_marker = sum(1 for line in non_empty_lines if line.lower().strip().startswith(('conclusion:', 'answer:', 'therefore,', 'in summary:')))
        
        structure_elements = numbered_lines + bullet_points + (conclusion_marker * 2)
        structure_ratio = min(1.0, structure_elements / (len(non_empty_lines) * 0.5))
        
        return 0.2 + structure_ratio * 0.8
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]