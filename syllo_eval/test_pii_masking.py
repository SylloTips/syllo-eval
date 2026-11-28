import os
import random
import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from pydantic import SecretStr

from syllo_eval.model import (
  GroundTruth,
  MetricComputation,
  MetricComputationStatus,
  MetricTargetingMode,
  Sample,
  Span,
)
from syllo_eval.pii_masking import PiiMasker
from syllo_eval.settings import PiiMaskingSettings
from syllo_eval.trace_semantics import (
  Answer,
  ExecutionStep,
  LlmUsage,
  PlannedStep,
  PlanningData,
  PlanSnapshot,
  RetrievalItem,
  RetrievalResult,
  SpanSemantics,
)

KEY = 'a-test-key-of-thirty-two-chars!!'
EMAIL = 'jane.doe@example.com'
# An identifier that reads as a phone number when it appears in free text.
PHONE_LIKE_ID = '3331234567'


class TestPatterns(unittest.TestCase):
  def setUp(self) -> None:
    self.masker = PiiMasker()

  def test_each_kind_of_personal_data_becomes_its_label(self) -> None:
    cases = {
      'url': ['https://docs.example.com/page?user=jane@example.com', 'ftp://files.example.org/a'],
      'email': [EMAIL, 'j+news@mail.example.co.uk'],
      'iban': ['IT60 X054 2811 1010 0000 0123 456', 'IT60X0542811101000000123456', 'GB82 WEST 1234 5698 7654 32'],
      'it_tax_code': ['RSSMRA80A01H501U', 'rssmra80a01h501u', 'MRTMTT25D09F205Z'],
      'mac': ['00:1A:2B:3C:4D:5E', '00-1a-2b-3c-4d-5e', '001a.2b3c.4d5e'],
      'ip': ['192.168.1.10', '2001:db8::1', 'fe80::1ff:fe23:4567:890a'],
      'card': ['4111 1111 1111 1111', '4111-1111-1111-1111', '5500000000000004', '378282246310005'],
      'ssn': ['123-45-6789'],
      'it_vat': ['IT12345678903', '12345678903'],
      'phone': [
        '+39 333 1234567',
        '+39 02 1234 5678',
        '0039 333 1234567',
        '3331234567',
        '(555) 123-4567',
        '+1 (555) 123-4567',
        '555.123.4567',
      ],
      'handle': ['@jane_doe'],
    }
    for category, values in cases.items():
      for value in values:
        with self.subTest(category=category, value=value):
          self.assertEqual(
            self.masker.mask_text(f'Contact: {value}, thanks.'), f'Contact: [{category.upper()}], thanks.'
          )

  def test_values_that_only_look_like_personal_data_stay(self) -> None:
    for text in [
      'Created 2026-11-27T10:15:30Z, updated 2026-11-27 10:15:30, due 27.11.2026 14.30, build 20261127.',
      'Years 2019-2024; 1.234.567 EUR; 12.345.678 EUR; 12,345,678.90 USD; 12345678.90 USD; 1 234 567 people.',
      'Unix times 1764237330 and 1764237330123; score 0.8333; 128000 tokens; amount 1234.5678.',
      'UUID 123e4567-e89b-12d3-a456-426614174000, span 7a3f9c2b1d4e5f60, id 1234567890123456.',
      'Times 10:15:30 and 3:2:1; C++ std::vector; loopback ::1; slices a[1::2] and x[::2].',
      'Failed checks: RSSMRA80A01H501X, IT60X0542811101000000123457, IT12345678904; ticket #123456; 1234567.',
    ]:
      with self.subTest(text=text):
        self.assertEqual(self.masker.mask_text(text), text)

  def test_a_valid_value_is_masked_without_what_follows_it(self) -> None:
    self.assertEqual(
      self.masker.mask_text('Card 4111 1111 1111 1111 123, IBAN ES91 2100 0418 4502 0005 1332 BANK.'),
      'Card [CARD] 123, IBAN [IBAN] BANK.',
    )

  def test_only_the_selected_categories_are_masked(self) -> None:
    masker = PiiMasker(categories=['email'])

    self.assertEqual(masker.mask_text(f'{EMAIL}, +39 333 1234567'), '[EMAIL], +39 333 1234567')

  def test_masking_masked_text_changes_nothing(self) -> None:
    fragments = [
      EMAIL,
      'https://x.org/a?u=jane@x.com',
      'IT60 X054 2811 1010 0000 0123 456',
      'RSSMRA80A01H501U',
      '00:1A:2B:3C:4D:5E',
      '192.168.1.10',
      '2001:db8::1',
      '4111 1111 1111 1111',
      '123-45-6789',
      'IT12345678903',
      '+39 333 1234567',
      '(555) 123-4567',
      '@jane_doe',
      '2026-11-27',
      '12.345.678',
      '[PHONE_12345678]',
      '123',
      'abc',
    ]
    separators = ['', ' ', '-', '.', ':', '_', '\n', ', ', '/', '@', '(', ')', '+']
    rng = random.Random(7)
    for masker in (PiiMasker(), PiiMasker(key=KEY)):
      for _ in range(1000):
        text = ''.join(rng.choice(fragments) + rng.choice(separators) for _ in range(rng.randint(1, 6)))
        masked = masker.mask_text(text)
        with self.subTest(text=text):
          self.assertEqual(masker.mask_text(masked), masked)


class TestLabels(unittest.TestCase):
  def test_without_a_key_every_value_of_a_kind_gets_the_same_label(self) -> None:
    self.assertEqual(PiiMasker().mask_text('Call 333 1234567 or 333 7654321.'), 'Call [PHONE] or [PHONE].')

  def test_with_a_key_each_value_gets_its_own_code(self) -> None:
    masker = PiiMasker(key=KEY)

    self.assertRegex(masker.mask_text('333 1234567'), r'^\[PHONE_[0-9a-f]{8}\]$')
    for spellings in [
      ('333 1234567', '333-123-4567', '3331234567'),
      ('Jane.Doe@Example.com', EMAIL),
      ('IT60 X054 2811 1010 0000 0123 456', 'IT60X0542811101000000123456'),
      ('IT12345678903', '12345678903'),
      ('00:1A:2B:3C:4D:5E', '001a.2b3c.4d5e'),
      ('2001:db8::1', '2001:0db8:0:0:0:0:0:1'),
    ]:
      with self.subTest(spellings=spellings):
        self.assertEqual(len({masker.mask_text(spelling) for spelling in spellings}), 1)
    self.assertNotEqual(masker.mask_text('333 1234567'), masker.mask_text('333 1234568'))
    self.assertNotEqual(
      masker.mask_text('333 1234567'), PiiMasker(key='another-key-of-sixteen').mask_text('333 1234567')
    )

  def test_settings_build_a_masker_only_when_enabled(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertIsNone(PiiMasker.from_settings(PiiMaskingSettings()))
      settings = PiiMaskingSettings(enabled=True, categories=('phone', 'email'), key=SecretStr(KEY))
      masker = PiiMasker.from_settings(settings)

    assert masker is not None
    self.assertEqual(masker.describe(), {'categories': ['email', 'phone'], 'keyed': True})
    self.assertRegex(masker.mask_text(EMAIL), r'^\[EMAIL_[0-9a-f]{8}\]$')


class TestModels(unittest.TestCase):
  def setUp(self) -> None:
    self.masker = PiiMasker()

  def test_span_free_text_is_masked_and_identifiers_stay(self) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    usage = LlmUsage(call_id=PHONE_LIKE_ID, model='judge-model')
    span = Span(
      external_id=PHONE_LIKE_ID,
      trace_id='trace-1',
      span_type='agent_root',
      name='agent',
      start_time=now,
      end_time=now,
      input_data={'question': f'My email is {EMAIL}', 'ids': [PHONE_LIKE_ID]},
      output_data=f'Write to {EMAIL}',
      metadata={'attributes.user.id': EMAIL},
      semantics=SpanSemantics(
        request=f'My email is {EMAIL}',
        answer=Answer(text=f'Write to {EMAIL}', citations=[PHONE_LIKE_ID]),
        retrieval=[
          RetrievalResult(
            kind='document',
            query=f'customer {EMAIL}',
            items=[
              RetrievalItem(
                id=PHONE_LIKE_ID,
                document_id=PHONE_LIKE_ID,
                content=f'Owner: {EMAIL}',
                title=EMAIL,
                location='https://crm.example.com/customers/1',
                attributes={'owner': EMAIL},
              )
            ],
          )
        ],
        planning=PlanningData(
          plans=[
            PlanSnapshot(
              id='plan-1',
              steps=[
                PlannedStep(
                  id=PHONE_LIKE_ID, operation='lookup', instruction=f'Find {EMAIL}', parameters={'email': EMAIL}
                )
              ],
            )
          ],
          executed_steps=[
            ExecutionStep(
              id='step-1',
              operation='lookup',
              instruction=f'Find {EMAIL}',
              plan_id='plan-1',
              planned_step_id=PHONE_LIKE_ID,
              input={'email': EMAIL},
              output=EMAIL,
              span_ids=[PHONE_LIKE_ID],
              attributes={'by': EMAIL},
            )
          ],
        ),
        usage=usage,
      ),
    )

    masked = self.masker.mask_model(span)

    self.assertNotIn(EMAIL, masked.model_dump_json())
    self.assertEqual(span.semantics.request, f'My email is {EMAIL}')
    self.assertEqual(masked.input_data, {'question': 'My email is [EMAIL]', 'ids': ['[PHONE]']})
    self.assertEqual(masked.external_id, PHONE_LIKE_ID)
    semantics = masked.semantics
    assert semantics.answer is not None and semantics.planning is not None
    self.assertEqual(semantics.answer.citations, [PHONE_LIKE_ID])
    item = semantics.retrieval[0].items[0]
    self.assertEqual((item.id, item.document_id, item.location), (PHONE_LIKE_ID, PHONE_LIKE_ID, '[URL]'))
    step = semantics.planning.plans[0].steps[0]
    self.assertEqual((step.id, step.operation, step.parameters), (PHONE_LIKE_ID, 'lookup', {'email': '[EMAIL]'}))
    executed = (semantics.planning.executed_steps or [])[0]
    self.assertEqual(
      (executed.planned_step_id, executed.span_ids, executed.operation), (PHONE_LIKE_ID, [PHONE_LIKE_ID], 'lookup')
    )
    self.assertEqual(semantics.usage, usage)

  def test_ground_truth_text_is_masked_and_identifiers_stay(self) -> None:
    sample_id = uuid4()
    cases: list[tuple[dict[str, Any], dict[str, Any]]] = [
      ({'expected_output': 'Call 333 1234567'}, {'expected_output': 'Call [PHONE]'}),
      ({'relevant_ids': [PHONE_LIKE_ID]}, {'relevant_ids': [PHONE_LIKE_ID]}),
      (
        {'claims': [{'id': PHONE_LIKE_ID, 'text': 'The phone is 333 1234567'}]},
        {'claims': [{'id': PHONE_LIKE_ID, 'text': 'The phone is [PHONE]'}]},
      ),
      (
        {'expected_plan': [{'operation': 'lookup', 'instruction': f'Find {EMAIL}', 'parameters': {'email': EMAIL}}]},
        {'expected_plan': [{'operation': 'lookup', 'instruction': 'Find [EMAIL]', 'parameters': {'email': '[EMAIL]'}}]},
      ),
    ]
    for value, expected in cases:
      ground_truth = GroundTruth(id=uuid4(), sample_id=sample_id, key='key', ground_truth_value=value)
      with self.subTest(value=value):
        masked = self.masker.mask_ground_truth(ground_truth)
        self.assertEqual((masked.id, masked.ground_truth_value), (ground_truth.id, expected))

  def test_sample_text_and_metric_reasoning_and_errors_are_masked(self) -> None:
    sample = Sample(id=uuid4(), dataset_id=uuid4(), input_prompt=f'I am {EMAIL}')
    computation = MetricComputation(
      id=uuid4(),
      evaluation_run_sample_id=uuid4(),
      metric='metric',
      targeting_mode=MetricTargetingMode.SINGLE,
      target_span_type='agent_root',
      span_ids=[PHONE_LIKE_ID],
      score=None,
      status=MetricComputationStatus.FAILED,
      reasoning=f'Quoted {EMAIL}',
      error_message=f'Bad input {EMAIL}',
      metadata={'retrieved_id': PHONE_LIKE_ID},
    )

    masked_sample = self.masker.mask_model(sample)
    masked_computation = self.masker.mask_model(computation)

    self.assertEqual((masked_sample.id, masked_sample.input_prompt), (sample.id, 'I am [EMAIL]'))
    self.assertIsNone(masked_sample.ground_truth_output)
    self.assertEqual(
      (masked_computation.reasoning, masked_computation.error_message), ('Quoted [EMAIL]', 'Bad input [EMAIL]')
    )
    self.assertEqual(masked_computation.span_ids, [PHONE_LIKE_ID])
    self.assertEqual(masked_computation.metadata, {'retrieved_id': PHONE_LIKE_ID})


if __name__ == '__main__':
  unittest.main()
