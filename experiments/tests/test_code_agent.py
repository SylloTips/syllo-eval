import json
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from agents.code_agent import ROOT_SPAN_NAME, CodeAgentTraceAdapter

TRACE_ID = 'c' * 32
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
SEARCH = {
  'call_id': 'call-1',
  'query': 'refund policy',
  'results': [
    {'rank': 1, 'id': 'doc-7', 'title': 'Refunds', 'text': 'Refunds take 5 days.'},
    {'rank': 2, 'id': 'doc-2', 'title': 'Returns', 'text': 'Returns are free.'},
  ],
}


def _record(span_id: str, name: str, kind: str, parent: str | None, **attributes: Any) -> dict[str, Any]:
  """A span as the Phoenix client returns it: attributes flattened under ``attributes.``."""
  second = len(span_id)
  return {
    'name': name,
    'status_code': 'OK',
    'parent_id': parent,
    'start_time': T0 + timedelta(seconds=second),
    'end_time': T0 + timedelta(seconds=second + 1),
    'context.span_id': span_id,
    'attributes.openinference.span.kind': kind,
    **{f'attributes.{key}': value for key, value in attributes.items()},
  }


def _trace(answer: str = '42') -> list[dict[str, Any]]:
  tool = {'tool.name': 'search_knowledge_base', 'input.value': json.dumps({'query': 'refund policy'})}
  return [
    _record('r', ROOT_SPAN_NAME, 'AGENT', None, **{'input.value': 'How long do refunds take?', 'output.value': answer}),
    _record('run', 'CodeAgent.run', 'AGENT', 'r'),
    _record('step', 'Step 1', 'CHAIN', 'run'),
    _record('llm', 'LiteLLMModel.generate', 'LLM', 'step', **{'llm.token_count.prompt': 100}),
    _record('ok', 'MCPAdaptTool', 'TOOL', 'step', **tool, **{'output.value': json.dumps(SEARCH)}),
    _record('fail', 'MCPAdaptTool', 'TOOL', 'step', **tool, **{'output.value': 'The search failed: timeout'}),
    _record('final', 'FinalAnswerTool', 'TOOL', 'step', **{'tool.name': 'final_answer', 'output.value': answer}),
  ]


class CodeAgentTraceAdapterTest(unittest.TestCase):
  def test_the_root_answers_and_each_search_is_a_ranked_retrieval_of_the_question(self) -> None:
    result = CodeAgentTraceAdapter().normalize(TRACE_ID, _trace())
    spans = {span.external_id: span for span in result.spans}

    root = spans['r']
    self.assertEqual(root.span_type, 'agent_root')
    self.assertEqual(root.semantics.request, 'How long do refunds take?')
    assert root.semantics.answer is not None
    self.assertEqual(root.semantics.answer.text, '42')

    search = spans['ok']
    self.assertEqual(search.span_type, 'retrieval')
    self.assertEqual(search.semantics.request, 'How long do refunds take?')
    [retrieval] = search.semantics.retrieval
    self.assertEqual((retrieval.kind, retrieval.stage, retrieval.query), ('document', 'selected', 'refund policy'))
    self.assertEqual([item.id for item in retrieval.items], ['doc-7', 'doc-2'])
    self.assertEqual(retrieval.items[0].content, 'Refunds take 5 days.')

    self.assertEqual((spans['fail'].span_type, spans['final'].span_type), ('tool', 'tool'))
    self.assertEqual(spans['llm'].semantics.usage.input_tokens, 100)  # type: ignore[union-attr]
    self.assertEqual(result.trace.adapter, 'smolagents:1')

  def test_a_trace_without_the_callers_root_is_refused(self) -> None:
    records = _trace()[1:]
    records[0]['parent_id'] = None
    with self.assertRaisesRegex(ValueError, ROOT_SPAN_NAME):
      CodeAgentTraceAdapter().normalize(TRACE_ID, records)


if __name__ == '__main__':
  unittest.main()
