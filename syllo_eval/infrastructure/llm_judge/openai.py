from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from syllo_eval.evaluation.judge import LlmJudgeRequest
from syllo_eval.infrastructure.llm_judge.base import LangChainLlmJudgeClient
from syllo_eval.settings import OpenAIJudgeSettings


class OpenAILlmJudgeClient(LangChainLlmJudgeClient):
  """Judge client backed by LangChain's OpenAI chat integration."""

  def __init__(
    self,
    config: OpenAIJudgeSettings,
    max_concurrent_requests: int,
    chat_model: Any | None = None,
  ):
    default_chat_model = chat_model or ChatOpenAI(
      model=config.model,
      api_key=SecretStr(config.api_key) if config.api_key is not None else None,
      base_url=config.base_url,
      timeout=config.timeout_seconds,
      max_retries=config.max_retries,
      use_responses_api=config.use_responses_api,
    )
    super().__init__(
      provider_name='openai',
      model=config.model,
      max_concurrent_requests=max_concurrent_requests,
      chat_model=default_chat_model,
      # json_mode works across OpenAI-compatible providers: DeepSeek rejects json_schema, and its reasoning models
      # reject the forced tool_choice of function calling. LangChain does not inject the schema, so the base class does.
      structured_output_method='json_mode',
      max_tokens_option='max_tokens',
      close=default_chat_model.root_async_client.close if chat_model is None else None,
    )

  def _model_options(self, request: LlmJudgeRequest) -> dict[str, Any]:
    options = super()._model_options(request)
    # gpt-5 reasoning models accept only their default temperature. LangChain drops any other value when it builds the
    # model, but not on a model copy.
    model = self.model.lower()
    if model.startswith('gpt-5') and 'chat' not in model:
      del options['temperature']
    return options
