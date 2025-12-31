import json
import time
from typing import Annotated, List, Optional

import PIL
import requests
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class OllamaService(BaseService):
    ollama_base_url: Annotated[
        str, "The base url to use for ollama.  No trailing slash."
    ] = "http://localhost:11434"
    ollama_model: Annotated[str, "The model name to use for ollama."] = (
        "llama3.2-vision"
    )
    
    ollama_api_key: Annotated[Optional[str], "The API key for Ollama if using a hosted service."] = None

    def process_images(self, images):
        image_bytes = [self.img_to_base64(img) for img in images]
        return image_bytes

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int = 600,
    ):
        url = f"{self.ollama_base_url}/api/generate"
        
        # Use provided timeout/retries or fall back to class defaults from BaseService
        exec_timeout = timeout or getattr(self, "timeout", 600)
        exec_retries = max_retries if max_retries is not None else getattr(self, "max_retries", 3)
        
        headers = {"Content-Type": "application/json"}
        # Priority: self.ollama_api_key (from config) -> Fallback (existing key)
        api_key = self.ollama_api_key 
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        schema = response_schema.model_json_schema()
        format_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

        image_bytes = self.format_image_for_llm(image)

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "images": image_bytes,
        }

        last_error = None
        for attempt in range(exec_retries + 1):
            try:
                if attempt > 0:
                    wait_time = getattr(self, "retry_wait_time", 3)
                    logger.info(f"Retrying Ollama inference (attempt {attempt}/{exec_retries}) in {wait_time}s...")
                    time.sleep(wait_time)

                response = requests.post(url, json=payload, headers=headers, timeout=exec_timeout)
                
                if response.status_code != 200:
                    try:
                        error_json = response.json()
                        err_msg = error_json.get("error", response.text)
                    except Exception:
                        err_msg = response.text
                    
                    # Enhanced Debugging for common Ollama errors
                    if response.status_code == 500:
                        if "llama runner process has terminated" in err_msg.lower():
                            logger.error(f"OLLAMA CRITICAL: Runner crashed loading '{self.ollama_model}'. "
                                         "This is usually due to missing tensors (architecture mismatch) or insufficient VRAM.")
                        elif "not found" in err_msg.lower():
                            logger.error(f"OLLAMA ERROR: Model '{self.ollama_model}' not found. Did you run 'ollama pull'?")
                    
                    response.raise_for_status()

                response_data = response.json()
                total_tokens = response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)

                if block:
                    block.update_metadata(llm_request_count=1, llm_tokens_used=total_tokens)

                data = response_data.get("response", "")
                try:
                    return json.loads(data)
                except json.JSONDecodeError as je:
                    logger.error(f"Ollama returned malformed JSON response: {data[:500]}...")
                    raise je

            except Exception as e:
                last_error = e
                # Break early on fatal model loading errors that retries won't fix
                if "terminated" in str(e).lower() or "not found" in str(e).lower():
                    logger.error(f"Fatal Ollama error detected. Aborting retries: {e}")
                    break
                logger.warning(f"Ollama inference attempt {attempt} failed: {e}")

        logger.error(f"Ollama inference permanently failed after {exec_retries} attempts. Last error: {last_error}")
        return {}
