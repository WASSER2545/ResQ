import requests
from copy import deepcopy
from openai import OpenAI
from typing import List, Dict, Optional

import google.generativeai as genai

class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def add(self, usage):
        if usage is None:
            return
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens

    def summary(self):
        return dict(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
        )

class ChatAgent:
    def __init__(self, provider: str, api_key: str, model: str,
                 base_url: Optional[str] = None,
                 enable_thinking: bool = False,
                 system_prompt: str = "You are a helpful assistant."):
        self.provider = provider
        self.model = model
        self.enable_thinking = enable_thinking
        self.token_tracker = TokenTracker()

        if provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif provider == "qwen":
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def add_system_message(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def chat(self, user_input: str, stream: bool = False) -> str:
        self.messages.append({"role": "user", "content": user_input})

        if self.provider in ["openai", "qwen"]:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=stream,
                extra_body={"enable_thinking": self.enable_thinking} if self.enable_thinking else None
            )

            if stream:
                reply = ""
                for chunk in completion:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end="", flush=True)
                    reply += delta
                print()
            else:
                reply = completion.choices[0].message.content
                print(reply)

                if hasattr(completion, "usage") and completion.usage:
                    self.token_tracker.add(completion.usage)
                    
        elif self.provider == "gemini":
            history_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in self.messages
            )
            response = self.client.generate_content(history_text, stream=stream)

            if stream:
                reply = ""
                for chunk in response:
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        reply += chunk.text
                print()
            else:
                reply = response.text
                print(reply)

        self.messages.append({"role": "assistant", "content": reply})
        return reply

def get_key(key_file):
        with open(key_file, "r") as f:
            keys = [x.strip() for x in f.readlines()]
        cur_key = deepcopy(keys[0])
        keys = keys[1:] + [cur_key]
        return cur_key