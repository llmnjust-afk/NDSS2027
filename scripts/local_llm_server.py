"""Lightweight local LLM server using transformers, exposing an OpenAI-compatible API.
Run: python -m scripts.local_llm_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000
"""
import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class OpenAICompatHandler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None
    model_name = ""

    def do_POST(self):
        if "/chat/completions" not in self.path:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        max_tokens = body.get("max_tokens", 512)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=body.get("temperature", 0.0) or 0.01,
                do_sample=body.get("temperature", 0.0) > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        resp = {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"total_tokens": inputs["input_ids"].shape[1] + len(text.split())},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, *args):
        pass  # suppress logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    OpenAICompatHandler.tokenizer = AutoTokenizer.from_pretrained(args.model)
    OpenAICompatHandler.model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    OpenAICompatHandler.model_name = args.model
    print(f"Model loaded. Serving on port {args.port}...")
    server = HTTPServer(("0.0.0.0", args.port), OpenAICompatHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
