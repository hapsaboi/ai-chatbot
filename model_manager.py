import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import requests
from config import ChatbotConfig
from utils import setup_logging

class ModelManager:
    def __init__(self, model_name: str = None, device: Optional[str] = None):
        self.logger = setup_logging(__name__)
        self.device = torch.device(device if device else ChatbotConfig.DEVICE)
        self.use_api = True
        # self.use_api = ChatbotConfig.USE_API if model_name is None else False
        self.model_name = model_name
        
        # Available local models config - used only if model_name is specified
        self.available_models = {
            'phi-2': "microsoft/phi-2",
            'gpt2': "gpt2",
            'falcon-7b': "tiiuae/falcon-7b",
            'mistralai/Pixtral-Large-Instruct-2411':"mistralai/Pixtral-Large-Instruct-2411",
            'HuggingFaceH4/zephyr-7b-beta':'HuggingFaceH4/zephyr-7b-beta'
        }
        
        self._initialize_models()
        
    def _initialize_models(self) -> None:
        """Initialize models based on configuration"""
        try:
            # Always initialize embedding model as it's lightweight
            self.logger.info("Loading embedding model")
            self.embedding_model = SentenceTransformer(
                ChatbotConfig.EMBEDDING_MODEL,
                device=self.device
            )
            
            # Only load local model if not using API
            if not self.use_api:
                self.logger.info("Loading local chat model")
                model_path = self.available_models.get(self.model_name, ChatbotConfig.LOCAL_CHAT_MODEL)
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to(self.device)
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.logger.info("Using API model")
                
            self.logger.info("Model initialization complete")
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def generate_response(self, context: str) -> str:
        """Generate response using either local model or API"""
        try:
            if self.use_api:
                return self._generate_api_response(context)
            else:
                return self._generate_local_response(context)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
            
    def _generate_api_response(self, context: str) -> str:
        """Generate response using Hugging Face API"""
        try:
            headers = {"Authorization": f"Bearer {ChatbotConfig.HF_TOKEN}"}
            
            # Get appropriate model path
            if self.model_name:
                model_path = self.available_models.get(self.model_name)
                if not model_path:
                    raise ValueError(f"Invalid model name: {self.model_name}")
            else:
                model_path = ChatbotConfig.CHAT_MODEL
                
            API_URL = f"https://api-inference.huggingface.co/models/{model_path}"
            
            # Format prompt
            prompt = (
                "You are a university information assistant. Using only the "
                "information below, answer the question accurately and concisely. "
                "Do not add or infer any additional information.\n\n"
                f"Information:\n{context}\n\n"
                "Answer:"
            )

            # Make API request
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 256,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1
                    }
                }
            )
            response.raise_for_status()
            
            # Extract generated text
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Clean up response
                if "Answer:" in generated_text:
                    generated_text = generated_text.split("Answer:")[-1].strip()
                return generated_text
            return "Error: Unexpected API response format"

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise

    def _generate_local_response(self, context: str) -> str:
        """Generate response using local model"""
        try:
            prompt = (
                "Instruct: You are a university information assistant. Using only the "
                "information below, answer the question accurately and concisely.\n\n"
                f"Information:\n{context}\n\n"
                "Response:"
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
                
            return response

        except Exception as e:
            self.logger.error(f"Error generating local response: {str(e)}")
            raise
            
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Generate embeddings for input texts"""
        try:
            return self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    @property
    def model_loaded(self) -> bool:
        """Check if required models are loaded"""
        if self.use_api:
            return hasattr(self, 'embedding_model')
        return all(hasattr(self, attr) for attr in ['model', 'tokenizer', 'embedding_model'])