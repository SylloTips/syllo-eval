import unittest

from syllo_eval.evaluation.judge import LlmJudgeRequest
from syllo_eval.evaluation.metrics.implementations.answer.correctness_judge import AnswerCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.plan.correctness_judge import PlanCorrectnessJudgeMetric
from syllo_eval.evaluation.metrics.implementations.rag.contextual_precision_judge import (
  ContextualPrecisionDocumentJudgeMetric,
)
from syllo_eval.evaluation.metrics.implementations.rag.contextual_recall_judge import (
  ContextualRecallDocumentJudgeMetric,
)
from syllo_eval.evaluation.metrics.test_support import Judge, span, truth
from syllo_eval.trace_semantics import Answer, ExecutionStep, PlanningData, RetrievalItem, RetrievalResult


def _documents(*contents: str) -> list[RetrievalResult]:
  items = [RetrievalItem(id=str(index), content=content) for index, content in enumerate(contents)]
  return [RetrievalResult(stage='selected', kind='document', items=items)]


def _judgment(statement: str) -> dict:
  return {'judgments': [{'statement': statement, 'attributable': True, 'reasoning': 'r'}]}


class PromptPrefixTest(unittest.IsolatedAsyncioTestCase):
  def assert_fixed_text_first(self, first: LlmJudgeRequest, second: LlmJudgeRequest) -> None:
    self.assertEqual(first.system_prompt, second.system_prompt)
    shared = [a == b for a, b in zip(first.user_prompt.split('\n\n'), second.user_prompt.split('\n\n'))]
    self.assertEqual(shared, sorted(shared, reverse=True), f'fixed text after sample text:\n{second.user_prompt}')

  async def test_requests_that_share_a_prefix_differ_only_after_it(self) -> None:
    score = {'score': 1, 'reasoning': 'r'}

    judge = Judge(score, score)
    answer = AnswerCorrectnessJudgeMetric(judge_client=judge)
    await answer.compute(span(answer=Answer(text='Paris')), truth(expected_output='Paris', rubric='Name the city'))
    await answer.compute(span(answer=Answer(text='Rome')), truth(expected_output='Rome', notes='Capital'))
    self.assert_fixed_text_first(*judge.requests)

    judge = Judge(score, score)
    plan = PlanCorrectnessJudgeMetric(judge_client=judge)
    for operation in ('search', 'answer'):
      executed = PlanningData(executed_steps=[ExecutionStep(id='1', operation=operation, instruction=operation)])
      await plan.compute(span(planning=executed), truth(expected_plan=[{'operation': operation}]))
    self.assert_fixed_text_first(*judge.requests)

    judge = Judge(*[{'judgments': [{'rank': rank, 'relevant': True, 'reasoning': 'r'}]} for rank in (1, 2)])
    await ContextualPrecisionDocumentJudgeMetric(judge_client=judge).compute(
      span(retrieval=_documents('Paris facts', 'Rome facts')), truth(expected_output='Paris')
    )
    self.assert_fixed_text_first(*judge.requests)

    judge = Judge({'claims': ['A', 'B']}, _judgment('A'), _judgment('B'))
    await ContextualRecallDocumentJudgeMetric(judge_client=judge).compute(
      span(retrieval=_documents('A facts', 'B facts')), truth(expected_output='A and B')
    )
    self.assert_fixed_text_first(*judge.requests[1:])


if __name__ == '__main__':
  unittest.main()
