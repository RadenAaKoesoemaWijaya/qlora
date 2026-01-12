import logging
import torch
import evaluate
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

class EvaluationEngine:
    """
    Engine untuk melakukan evaluasi model LLM menggunakan berbagai metrik:
    - Perplexity: Mengukur fluency
    - BERTScore: Mengukur semantic similarity
    - ROUGE: Mengukur structural similarity (n-gram overlap)
    - BLEU: Mengukur precision overlap
    """
    
    def __init__(self, base_model_id: str, adapter_path: str = None, device: str = "auto"):
        self.device = device
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
        # Load Metrics
        try:
            self.bert_score = evaluate.load("bertscore")
            self.rouge = evaluate.load("rouge")
            self.bleu = evaluate.load("bleu")
        except Exception as e:
            logger.warning(f"Failed to load evaluation metrics online, trying offline or skipping: {e}")
            # Fallback handling could be added here
            
    async def load_model(self):
        """Load model dan tokenizer"""
        logger.info(f"Loading model for evaluation: {self.base_model_id}")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load Adapter if provided
        if self.adapter_path:
            logger.info(f"Loading adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            
        self.model.eval()

    def calculate_perplexity(self, text_list: List[str]) -> float:
        """Hitung perplexity pada list text"""
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in text_list:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses untuk list prompts"""
        responses = []
        
        generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.1, # Low temp for reproducibility
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode dan ambil hanya bagian generated (buang prompt)
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_text.replace(prompt, "").strip()
            responses.append(response)
            
        return responses

    def evaluate_dataset(self, test_dataset: List[Dict[str, str]], limit: int = 20) -> Dict[str, Any]:
        """
        Evaluasi dataset test.
        Dataset harus memiliki format: [{'instruction': '...', 'output': '...'}]
        """
        if not self.model:
            raise RuntimeError("Model belum di-load. Panggil load_model() terlebih dahulu.")

        # Batasi sampel untuk efisiensi
        subset = test_dataset[:limit]
        
        prompts = [item.get("instruction", "") for item in subset]
        references = [item.get("output", "") for item in subset]
        
        # 1. Generate Predictions
        logger.info("Generating responses for evaluation...")
        predictions = self.generate_responses(prompts)
        
        # 2. Calculate Metrics
        logger.info("Calculating metrics...")
        
        # BERTScore
        bert_result = self.bert_score.compute(
            predictions=predictions, 
            references=references, 
            lang="en" # Sesuaikan bahasa jika perlu
        )
        avg_bert_f1 = sum(bert_result['f1']) / len(bert_result['f1'])
        
        # ROUGE
        rouge_result = self.rouge.compute(predictions=predictions, references=references)
        
        # BLEU
        bleu_result = self.bleu.compute(predictions=predictions, references=references)
        
        # Perplexity (dihitung dari referensi asli untuk melihat seberapa 'kaget' model terhadap ground truth)
        perplexity = self.calculate_perplexity(references)
        
        return {
            "bert_score_f1": avg_bert_f1,
            "rouge1": rouge_result['rouge1'],
            "rouge2": rouge_result['rouge2'],
            "rougeL": rouge_result['rougeL'],
            "bleu": bleu_result['bleu'],
            "perplexity": perplexity,
            "samples": [
                {
                    "prompt": p,
                    "prediction": pred,
                    "reference": ref
                } for p, pred, ref in zip(prompts, predictions, references[:3]) # Simpan 3 contoh
            ]
        }
