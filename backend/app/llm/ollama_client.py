"""
Robust Ollama client with retry logic, timeout handling, and fallback strategies.
Replaces direct Ollama calls with centralized, production-ready interface.
"""
import json
import time
import requests
from typing import Optional, Dict, Any
from app.utils.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class OllamaClient:
    """Production-ready Ollama client with error handling."""
    
    def __init__(self, base_url: str = None, model: str = None, timeout: int = 60):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout
        self.retry_count = 3
        self.retry_delay = 1
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.3,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> Optional[str]:
        """
        Generate text using Ollama with retry logic.
        
        Args:
            prompt: The prompt to generate from
            system: System prompt/context
            temperature: Sampling temperature (0.0 - 2.0)
            top_k: Top-K sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated text or None if failed
        """
        if not self.is_available():
            logger.warning("Ollama not available")
            return None
        
        for attempt in range(self.retry_count):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "num_predict": 2048,
                    },
                }
                
                if system:
                    payload["system"] = system
                
                logger.debug(f"Ollama request (attempt {attempt + 1}): {prompt[:100]}...")
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                result = response.json().get("response", "")
                logger.debug(f"Ollama response: {result[:100]}...")
                return result
                
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout on attempt {attempt + 1}/{self.retry_count}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    
            except requests.exceptions.ConnectionError:
                logger.warning(f"Ollama connection error on attempt {attempt + 1}/{self.retry_count}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Ollama error on attempt {attempt + 1}: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        
        logger.error("Ollama failed after all retries")
        return None
    
    def json_response(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate JSON response from Ollama with validation.
        
        Returns:
            Parsed JSON dict or None if parsing failed
        """
        response_text = self.generate(prompt, system, temperature)
        
        if not response_text:
            return None
        
        # Try to extract JSON from response
        try:
            # Try direct parsing first
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
            
            logger.error(f"Failed to parse JSON from Ollama response: {response_text[:200]}")
            return None


# Global instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create global Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
