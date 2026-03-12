import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from MapOMOP.llm_client import (
    LLMProvider,
    _default_max_tokens_for_model,
    _escape_prompt_content,
    _normalize_prompt_role,
    get_env_config,
    get_llm_client,
    normalize_model_name,
    normalize_provider,
)


SMOKE_TEST_TARGETS = [
    {
        "label": "default_openai",
        "provider": LLMProvider.OPENAI,
        "model": "gpt-5-mini-2025-08-07",
    },
    {
        "label": "together_gpt_oss_20b",
        "provider": LLMProvider.TOGETHER,
        "model": "gpt_oss_20b",
    },
    {
        "label": "together_mistral_small_24b",
        "provider": LLMProvider.TOGETHER,
        "model": "mistral_small_24b",
    },
    {
        "label": "together_llama4_maverick",
        "provider": LLMProvider.TOGETHER,
        "model": "llama4_maverick",
    },
]


class LLMClientConfigTest(unittest.TestCase):
    def test_normalize_provider_aliases(self):
        self.assertEqual(normalize_provider("openai"), LLMProvider.OPENAI)
        self.assertEqual(normalize_provider("together-ai"), LLMProvider.TOGETHER)
        self.assertEqual(normalize_provider("unknown-provider"), LLMProvider.OPENAI)

    def test_provider_specific_env_overrides_common_env(self):
        env = {
            "LLM_PROVIDER": "together",
            "LLM_MODEL": "shared-model",
            "LLM_BASE_URL": "http://shared/v1",
            "LLM_TEMPERATURE": "0.7",
            "LLM_MAX_TOKENS": "2048",
            "TOGETHER_MODEL": "gpt_oss_20b",
            "TOGETHER_BASE_URL": "https://api.together.xyz/v1",
            "TOGETHER_TEMPERATURE": "0.2",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_env_config()

        self.assertEqual(config["provider"], LLMProvider.TOGETHER)
        self.assertEqual(config["model"], "openai/gpt-oss-20b")
        self.assertEqual(config["base_url"], "https://api.together.xyz/v1")
        self.assertEqual(config["temperature"], 0.2)
        self.assertEqual(config["top_p"], 1.0)
        self.assertEqual(config["max_tokens"], 2048)

    def test_gpt5_model_defaults_to_128k_tokens(self):
        env = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-5-mini-2025-08-07",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_env_config()

        self.assertEqual(config["provider"], LLMProvider.OPENAI)
        self.assertEqual(config["model"], "gpt-5-mini-2025-08-07")
        self.assertEqual(config["max_tokens"], 128000)
        self.assertEqual(_default_max_tokens_for_model("gpt-5-mini-2025-08-07"), 128000)

    def test_together_provider_reads_provider_specific_env(self):
        env = {
            "LLM_PROVIDER": "together",
            "TOGETHER_MODEL": "gpt_oss_20b",
            "TOGETHER_BASE_URL": "https://api.together.xyz/v1",
            "TOGETHER_API_KEY": "together-key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_env_config()

        self.assertEqual(config["provider"], LLMProvider.TOGETHER)
        self.assertEqual(config["model"], "openai/gpt-oss-20b")
        self.assertEqual(config["base_url"], "https://api.together.xyz/v1")
        self.assertEqual(config["api_key"], "together-key")

    def test_explicit_provider_does_not_inherit_other_provider_shared_model(self):
        env = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-5-mini-2025-08-07",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_env_config(provider="together")

        self.assertEqual(config["provider"], LLMProvider.TOGETHER)
        self.assertEqual(config["model"], "openai/gpt-oss-20b")
        self.assertEqual(config["base_url"], "https://api.together.xyz/v1")

    def test_explicit_overrides_bypass_singleton(self):
        client_a = get_llm_client(
            provider="openai",
            model="gpt-5-mini-2025-08-07",
            base_url="https://api.openai.com/v1",
            api_key="dummy",
        )
        client_b = get_llm_client(
            provider="together",
            model="mistral_small_24b",
            base_url="https://api.together.xyz/v1",
            api_key="dummy",
        )

        self.assertEqual(client_a.provider, LLMProvider.OPENAI)
        self.assertEqual(client_a.model, "gpt-5-mini-2025-08-07")
        self.assertEqual(client_b.provider, LLMProvider.TOGETHER)
        self.assertEqual(client_b.model, "mistralai/Mistral-Small-24B-Instruct-2501")
        self.assertIsNot(client_a, client_b)

    def test_together_model_aliases_expand_to_canonical_names(self):
        self.assertEqual(
            normalize_model_name(LLMProvider.TOGETHER, "gpt_oss_20b"),
            "openai/gpt-oss-20b",
        )
        self.assertEqual(
            normalize_model_name(LLMProvider.TOGETHER, "mistral_small_24b"),
            "mistralai/Mistral-Small-24B-Instruct-2501",
        )
        self.assertEqual(
            normalize_model_name(LLMProvider.TOGETHER, "llama4_maverick"),
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        )

    def test_prompt_content_is_escaped_for_langchain_templates(self):
        text = '{"rankings": [{"concept_id": "1"}]}'
        escaped = _escape_prompt_content(text)
        self.assertEqual(escaped, '{{"rankings": [{{"concept_id": "1"}}]}}')

    def test_prompt_roles_are_normalized_for_langchain(self):
        self.assertEqual(_normalize_prompt_role("user"), "human")
        self.assertEqual(_normalize_prompt_role("assistant"), "ai")
        self.assertEqual(_normalize_prompt_role("system"), "system")


def _build_smoke_client(target: dict):
    if target["provider"] == LLMProvider.OPENAI:
        config = get_env_config(provider=LLMProvider.OPENAI)
        return get_llm_client(
            provider=LLMProvider.OPENAI,
            model=config["model"],
            base_url=config["base_url"],
            api_key=config["api_key"],
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
        )

    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        raise RuntimeError("TOGETHER_API_KEY is required for Together smoke tests")

    together_base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
    return get_llm_client(
        provider=LLMProvider.TOGETHER,
        model=target["model"],
        base_url=together_base_url,
        api_key=together_api_key,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
    )


def _smoke_messages(label: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are running a connectivity smoke test. Be brief.",
        },
        {
            "role": "user",
            "content": (
                f"Reply in one short line only: OK | {label}. "
                "Do not add explanations or markdown."
            ),
        },
    ]


def run_smoke_tests() -> int:
    failures = 0

    for target in SMOKE_TEST_TARGETS:
        label = target["label"]
        print(f"\n=== {label} ===")
        try:
            client = _build_smoke_client(target)
            info = client.get_info()
            print(
                f"provider={info['provider']} model={info['model']} "
                f"base_url={info['base_url']} max_tokens={info['max_tokens']}"
            )

            response = client.chat_completion(
                messages=_smoke_messages(label),
                temperature=0.0,
                top_p=1.0,
                max_tokens=64,
                json_mode=False,
            )
            if not response or not response.strip():
                failures += 1
                print("FAIL: empty response")
                continue

            print(f"response={response.strip()}")
        except Exception as exc:
            failures += 1
            print(f"FAIL: {exc}")

    if failures:
        print(f"\nSmoke tests finished with {failures} failure(s).")
        return 1

    print("\nSmoke tests finished successfully.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="LLM client unit tests and live smoke tests"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run config/unit tests instead of live model smoke tests.",
    )
    args = parser.parse_args()

    if args.unit:
        unittest.main(argv=[sys.argv[0]])
        return

    raise SystemExit(run_smoke_tests())


if __name__ == "__main__":
    main()
