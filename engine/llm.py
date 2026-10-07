"""One function to call whichever LLM the workspace is configured for."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .workspace import Workspace


def call_llm(ws: Workspace, history, user_message, system, json_mode=False):
    provider = os.getenv("LLM_PROVIDER") or ws.config["llm"]["provider"]
    model = os.getenv("LLM_MODEL") or ws.config["llm"]["model"]
    messages = history + [{"role": "user", "content": user_message}]

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(model=model, max_tokens=900, system=system, messages=messages)
        return resp.content[0].text

    from openai import OpenAI
    if provider == "groq":
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
    else:
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)

    kwargs = dict(model=model, messages=[{"role": "system", "content": system}] + messages)
    if provider == "groq" and model.startswith("openai/gpt-oss"):
        kwargs["max_completion_tokens"] = 2000   # reasoning tokens count toward the limit
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["max_tokens"] = 900
        kwargs["temperature"] = 0.2
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content