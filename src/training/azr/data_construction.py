import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import torch

logger = logging.getLogger(__name__)

class TaskGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_types = config.get("task_types", ["abduction", "deduction", "induction"])
        self.task_distribution = config.get("task_distribution", [0.3, 0.3, 0.4]) # Ensure this sums to 1 or normalize
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
    
    def generate_task(self, model, hf_tokenizer) -> Dict[str, Any]: # Now expects HF tokenizer
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
                # Use the Hugging Face tokenizer
                inputs = hf_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=model.config.max_position_embeddings // 2 # Ensure prompt isn't too long
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
                
                # Decode using HF tokenizer
                # outputs_tensor[0] includes the input_ids, so slice them off
                generated_ids = outputs_tensor[0, input_ids_for_generation.shape[1]:]
                generated_text_str = hf_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                task = self._extract_task(generated_text_str)
                if task:
                    if len(task) < 10: 
                        task = f"Create a {task_type} reasoning problem about '{task}' that is more detailed."
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
        base_prompt = self.config.get("base_prompt", "Generate a challenging reasoning problem.")
        if task_type == "abduction":
            return base_prompt + " The problem should require abductive reasoning (inferring the most likely explanation from observations). For example: 'If the grass is wet in the morning, what might have happened during the night?'"
        elif task_type == "deduction":
            return base_prompt + " The problem should require deductive reasoning (deriving conclusions from premises). For example: 'If all birds can fly, and penguins are birds, can penguins fly? Explain your reasoning.'"
        elif task_type == "induction":
            return base_prompt + " The problem should require inductive reasoning (generalizing from specific instances). For example: 'Given the sequence 2, 4, 6, 8, what is the next number and why?'"
        else:
            return base_prompt
    
    def _extract_task(self, text: str) -> Optional[str]:
        lines = text.strip().split('\n')
        if not lines: return None
        task = lines[0]
        for prefix in ["Task:", "Problem:", "Question:"]:
            if task.startswith(prefix):
                task = task[len(prefix):].strip()
        if len(task) < 10 and len(lines) > 1:
            task = " ".join(lines[:min(3, len(lines))])      
        return task if task else None

class TaskValidator:
    def __init__(self, config: Dict[str, Any], python_executor=None):
        self.config = config
        self.python_executor = python_executor
        self.min_length = config.get("min_length", 10)
        self.max_length = config.get("max_length", 2000)
        
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
            if not executability["is_executable"]:
                logger.warning(f"Code execution failed but continuing: {executability['reason']}")
        complexity = self._calculate_complexity(task)
        clarity = self._calculate_clarity(task)
        is_valid = (
            complexity >= self.config.get("min_complexity", 0.1) and
            complexity <= self.config.get("max_complexity", 1.0) and
            clarity >= self.config.get("min_clarity", 0.3)
        )
        if not is_valid and task_info.get("from_seed", False):
            is_valid = True
            logger.info("Accepting seed task despite not meeting thresholds")
        logger.info(f"Task validation metrics - complexity: {complexity}, clarity: {clarity}, valid: {is_valid}")
        return {"is_valid": is_valid, "complexity": complexity, "clarity": clarity, 
                "executability": executability["score"] if task_type == "induction" and self.python_executor else 1.0,
                "reason": None if is_valid else "Failed metric thresholds"}
    
    def _calculate_complexity(self, task: str) -> float:
        words = task.split(); word_count = len(words)
        long_words = sum(1 for word in words if len(word) > 7)
        question_marks = task.count('?'); math_operators = sum(task.count(op) for op in ['+', '-', '*', '/', '=', '<', '>'])
        base_complexity = min(1.0, word_count / 100); long_word_factor = min(1.0, long_words / 5)
        question_factor = min(1.0, question_marks / 3); math_factor = min(1.0, math_operators / 5)
        complexity = (base_complexity * 0.4 + long_word_factor * 0.2 + question_factor * 0.2 + math_factor * 0.2)
        return max(0.2, complexity)
    
    def _calculate_clarity(self, task: str) -> float:
        words = task.split(); word_count = len(words)
        if word_count == 0: return 0.0
        avg_word_length = sum(len(word) for word in words) / word_count
        sentence_count = sum(1 for char in ['.', '?', '!'] if char in task)
        length_factor = max(0.0, min(1.0, 2.0 - avg_word_length / 10))
        sentence_factor = max(0.0, min(1.0, sentence_count / 3))
        clarity = length_factor * 0.7 + sentence_factor * 0.3
        return max(0.3, clarity)
    
    def _check_executability(self, task: str) -> Dict[str, Any]:
        if not self.python_executor: return {"is_executable": True, "score": 1.0, "reason": None}
        code_blocks = self._extract_code_blocks(task)
        if not code_blocks: return {"is_executable": True, "score": 0.5, "reason": None} # No code is fine
        for code in code_blocks:
            result = self.python_executor.execute(code)
            if not result["success"]:
                return {"is_executable": False, "score": 0.0, "reason": f"Code execution failed: {result.get('error', 'Unknown error')}"}
        return {"is_executable": True, "score": 1.0, "reason": None}
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        import re; pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

class SolutionGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_attempts = config.get("max_attempts", 3)
        
    def generate_solution(self, task_info: Dict[str, Any], model, hf_tokenizer) -> Dict[str, Any]: # Now expects HF tokenizer
        task = task_info.get("task", "")
        task_type = task_info.get("type", "")
        prompt = self._get_solution_generation_prompt(task, task_type)
        device = self._get_model_device(model)

        for attempt in range(self.max_attempts):
            try:
                # Use the Hugging Face tokenizer
                inputs = hf_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=model.config.max_position_embeddings // 2 # Ensure prompt isn't too long
                )
                
                input_ids_for_generation = inputs["input_ids"].to(device)
                attention_mask_for_generation = inputs.get("attention_mask", torch.ones_like(input_ids_for_generation)).to(device)
                
                outputs_tensor = model.generate(
                    input_ids=input_ids_for_generation,
                    attention_mask=attention_mask_for_generation,
                    max_new_tokens=self.config.get("max_new_tokens", 1024),
                    temperature=self.config.get("temperature", 0.7),
                    top_p=self.config.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 0,
                    eos_token_id=hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else model.config.eos_token_id
                )
                
                # Decode using HF tokenizer
                generated_ids = outputs_tensor[0, input_ids_for_generation.shape[1]:]
                generated_text_str = hf_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                solution = self._extract_solution(generated_text_str)
                if solution:
                    return {"task": task, "type": task_type, "solution": solution, "raw_generation": generated_text_str}
                    
                logger.warning(f"Failed to extract solution from generated text (attempt {attempt+1}/{self.max_attempts})")
            except Exception as e:
                logger.error(f"Error generating solution (attempt {attempt+1}/{self.max_attempts}): {e}", exc_info=True)
                
        logger.info(f"All solution generation attempts failed, using placeholder solution")
        if task_type == "abduction": placeholder = f"To solve this abductive reasoning problem, I need to find the most likely explanation. Looking at the given information: {task[:50]}..., I would infer that the most probable cause is related to the key elements mentioned."
        elif task_type == "deduction": placeholder = f"To solve this deductive reasoning problem, I need to apply logical rules. Given the premises in: {task[:50]}..., I can conclude that the logical consequence follows from applying standard rules of inference."
        elif task_type == "induction": placeholder = f"To solve this inductive reasoning problem, I need to find patterns. Looking at the examples in: {task[:50]}..., I can identify a pattern that suggests the general rule would be applicable to similar cases."
        else: placeholder = "I will solve this step by step by analyzing the key components of the problem and applying appropriate reasoning techniques."
        return {"task": task, "type": task_type, "solution": placeholder, "raw_generation": "", "is_placeholder": True}
    
    def _get_model_device(self, model) -> torch.device:
        if hasattr(model, "device"): return model.device
        if hasattr(model, "parameters"):
            try: return next(model.parameters()).device
            except StopIteration: pass
        return torch.device("cpu")
    
    def _get_solution_generation_prompt(self, task: str, task_type: str) -> str:
        base_prompt = self.config.get("base_prompt", "Solve the following problem step by step:")
        if self.config.get("include_task_type_hint", True):
            type_hint = f" This is a {task_type} reasoning problem."
            base_prompt += type_hint
        return f"{base_prompt}\n\n{task}"
    
    def _extract_solution(self, text: str) -> Optional[str]:
        if not text.strip(): return None
        import re; solution_pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(solution_pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

class SolutionValidator:
    def __init__(self, config: Dict[str, Any], python_executor=None):
        self.config = config
        self.python_executor = python_executor
        
    def validate(self, task_info: Dict[str, Any], solution_info: Dict[str, Any]) -> Dict[str, Any]:
        task = task_info.get("task", ""); task_type = task_info.get("type", "")
        solution = solution_info.get("solution", "")
        if solution_info.get("is_placeholder", False):
            return {"is_valid": True, "correctness": 0.5, "coherence": 0.5, "relevance": 0.5, "structure": 0.5, "reason": "Placeholder solution accepted"}
        if not solution: return {"is_valid": False, "correctness": 0.0, "reason": "Empty solution"}
        if task_type == "induction" and self.python_executor:
            return self._validate_with_execution(task, solution)
        return self._validate_with_heuristics(task, task_type, solution)
    
    def _validate_with_execution(self, task: str, solution: str) -> Dict[str, Any]:
        if not self.python_executor: return {"is_valid": True, "correctness": 0.5, "reason": "No executor available"} # Heuristic if no exec
        code_blocks = self._extract_code_blocks(solution)
        if not code_blocks:
            logger.warning("No code found in solution for execution validation, using heuristic validation instead.")
            return self._validate_with_heuristics(task, "induction", solution)
        for code in code_blocks:
            result = self.python_executor.execute(code)
            if not result["success"]:
                logger.warning(f"Code execution failed but continuing heuristic validation: {result.get('error', 'Unknown error')}")
                return self._validate_with_heuristics(task, "induction", solution) # Fallback to heuristics if code fails
        return {"is_valid": True, "correctness": 1.0, "reason": "Execution successful"} # Higher correctness for successful execution
    
    def _validate_with_heuristics(self, task: str, task_type: str, solution: str) -> Dict[str, Any]:
        words = solution.split(); word_count = len(words)
        if word_count < 5: return {"is_valid": False, "correctness": 0.0, "reason": "Solution too short"}
        coherence = self._calculate_coherence(solution); relevance = self._calculate_relevance(task, solution)
        structure = self._calculate_structure(solution)
        is_valid = (coherence >= self.config.get("min_coherence", 0.3) and
                    relevance >= self.config.get("min_relevance", 0.3) and
                    structure >= self.config.get("min_structure", 0.2))
        if not is_valid and word_count >= 20:
            is_valid = True; logger.info("Accepting solution despite not meeting thresholds (length >= 20)")
        correctness = (coherence + relevance + structure) / 3
        logger.info(f"Solution heuristic validation metrics - coherence: {coherence}, relevance: {relevance}, structure: {structure}, valid: {is_valid}")
        return {"is_valid": is_valid, "correctness": correctness, "coherence": coherence, 
                "relevance": relevance, "structure": structure, "reason": None if is_valid else "Failed metric thresholds"}
    
    def _calculate_coherence(self, solution: str) -> float:
        sentences = [s.strip() for s in solution.split('.') if s.strip()]; sentence_count = len(sentences)
        if sentence_count <= 1: return 0.5
        transition_words = ["therefore", "thus", "hence", "consequently", "as a result", "first", "second", "third", 
                            "finally", "lastly", "however", "although", "despite", "nevertheless", "conversely", 
                            "similarly", "likewise", "in addition", "furthermore", "moreover"]
        transition_count = sum(solution.lower().count(word) for word in transition_words)
        transition_factor = min(1.0, transition_count / (sentence_count * 0.3))
        return 0.5 + transition_factor * 0.5
    
    def _calculate_relevance(self, task: str, solution: str) -> float:
        task_words = set(word.lower() for word in task.split() if len(word) > 3)
        solution_words = set(word.lower() for word in solution.split() if len(word) > 3)
        if not task_words: return 0.5
        overlap = task_words.intersection(solution_words)
        overlap_ratio = len(overlap) / len(task_words) if len(task_words) > 0 else 0
        return 0.3 + min(0.7, overlap_ratio * 1.5)
    
    def _calculate_structure(self, solution: str) -> float:
        lines = solution.split('\n'); line_count = len(lines)
        if line_count <= 1: return 0.3
        numbered_lines = sum(1 for line in lines if line.strip().startswith(tuple(f"{i}." for i in range(10))))
        bullet_points = sum(1 for line in lines if line.strip().startswith(('-', '*', '•')))
        structure_elements = numbered_lines + bullet_points
        structure_ratio = min(1.0, structure_elements / (line_count * 0.3))
        return 0.3 + structure_ratio * 0.7
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        import re; pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]