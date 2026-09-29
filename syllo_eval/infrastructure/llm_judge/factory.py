from syllo_eval.evaluation.judge import LlmJudgeClient
from syllo_eval.infrastructure.exceptions import ConfigurationError
from syllo_eval.infrastructure.llm_judge.gemini import GeminiLlmJudgeClient
from syllo_eval.infrastructure.llm_judge.openai import OpenAILlmJudgeClient
from syllo_eval.settings import LlmJudgeSettings


def build_llm_judge_client(config: LlmJudgeSettings) -> LlmJudgeClient | None:
  """Build the configured LLM judge client, or return None when disabled."""
  if config.provider is None:
    return None
  if config.provider == 'openai':
    return OpenAILlmJudgeClient(config=config.openai, max_concurrent_requests=config.max_concurrent_requests)
  if config.provider == 'gemini':
    return GeminiLlmJudgeClient(config=config.gemini, max_concurrent_requests=config.max_concurrent_requests)
  raise ConfigurationError('llm_judge', 'provider', f'unsupported provider: {config.provider}')
