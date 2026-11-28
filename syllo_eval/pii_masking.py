"""Masking personal data before Syllo-eval stores it or a metric sends it to a model.

The approach follows potato's PII redaction (github.com/davidjurgens/potato): each match becomes a label such as
`[EMAIL]`, so a reader still knows that something was there. With a key, the label also carries a code of the value,
such as `[EMAIL_3fa2c19b]`: equal values get equal codes, so a judge can still tell a wrong phone number from the right
one. Labels match no pattern, so masking masked text changes nothing.

The patterns err toward masking: a long digit run, such as an order number, reads as a phone number. They miss names,
street addresses and other personal details written in prose, so masking is not a compliance tool.
"""

import hashlib
import hmac
import ipaddress
import re
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, get_args

from pydantic import BaseModel

from syllo_eval.model import GroundTruth, MetricComputation, Sample, Span
from syllo_eval.settings import PiiCategory, PiiMaskingSettings
from syllo_eval.trace_semantics import (
  Answer,
  ExecutionStep,
  PlannedStep,
  PlanningData,
  PlanSnapshot,
  RetrievalItem,
  RetrievalResult,
  SpanSemantics,
)

CATEGORIES: tuple[PiiCategory, ...] = get_args(PiiCategory)

# A number must not continue a word, a decimal number or a hyphenated identifier such as a UUID.
_NUMBER_START = r'(?<![\w+-])(?<!\d\.)'
_NUMBER_END = r'(?![\w-]|\.\d)'
_HEX = '[0-9A-Fa-f]'
_DATE = re.compile(
  r'\d{4}([.-])\d\d\1\d\d(?!\d)|\d\d([.-])\d\d\2\d{4}(?!\d)|(?:19|20)\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])$'
)
# The value of each character at an odd position of an Italian tax code; digits count as the letters A to J.
_TAX_CODE_ODD_VALUES = (1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23)


def _luhn(text: str) -> bool:
  """The checksum of card numbers and Italian VAT numbers."""
  total = 0
  for position, char in enumerate(reversed(re.sub(r'\D', '', text))):
    digit = int(char) * (2 if position % 2 else 1)
    total += digit - 9 if digit > 9 else digit
  return total % 10 == 0


def _valid_iban(text: str) -> bool:
  compact = text.replace(' ', '')
  return int(''.join(str(int(char, 36)) for char in compact[4:] + compact[:4])) % 97 == 1


def _valid_it_tax_code(text: str) -> bool:
  code = text.upper()
  total = 0
  for position, char in enumerate(code[:15]):
    value = int(char) if char.isdigit() else ord(char) - ord('A')
    total += _TAX_CODE_ODD_VALUES[value] if position % 2 == 0 else value
  return code[15] == chr(ord('A') + total % 26)


def _valid_ip(text: str) -> bool:
  try:
    address = ipaddress.ip_address(text)
  except ValueError:
    return False
  # Short IPv6 forms such as `::1` or `1::2` read like code; a written address has a group of three or more digits.
  return address.version == 4 or any(len(group) >= 3 for group in text.split(':'))


def _valid_phone(text: str) -> bool:
  lengths = [len(run) for run in re.findall(r'\d+', text)]
  return (
    8 <= sum(lengths) <= 15
    # Not a date (2026-11-27, 27.11.2026, 20261127), a pair of years (2019-2024), a Unix time in seconds or
    # milliseconds (1764237330), a decimal number (12345678.90) or an amount in thousands (12.345.678).
    and _DATE.match(text) is None
    and lengths != [4, 4]
    and not (text.isdigit() and text[0] == '1' and len(text) in (10, 13))
    and re.search(r'\.\d{1,2}$', text) is None
    and not (text[0] in '123456789' and len(lengths) > 2 and lengths[0] <= 3 and set(lengths[1:]) == {3})
  )


_RULES: dict[PiiCategory, tuple[re.Pattern[str], Callable[[str], bool] | None]] = {
  # The closing punctuation of a sentence does not belong to the URL.
  'url': (re.compile(r'\b(?:https?|ftp)://[^\s<>"\']*[^\s<>"\'.,;:!?)\]}]', re.ASCII), None),
  'email': (
    re.compile(r'(?<![\w.%+-])[\w.%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,63}', re.ASCII),
    None,
  ),
  'iban': (re.compile(r'\b[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b', re.ASCII), _valid_iban),
  # Codice fiscale: name letters, birth date and place, check letter; letters may replace digits (omocodia).
  'it_tax_code': (
    re.compile(
      r'\b[A-Z]{6}[0-9LMNP-V]{2}[ABCDEHLMPRST][0-9LMNP-V]{2}[A-Z][0-9LMNP-V]{3}[A-Z]\b', re.ASCII | re.IGNORECASE
    ),
    _valid_it_tax_code,
  ),
  'mac': (
    re.compile(
      rf'(?<![\w:.-])(?:{_HEX}{{2}}([:-])(?:{_HEX}{{2}}\1){{4}}{_HEX}{{2}}|{_HEX}{{4}}\.{_HEX}{{4}}\.{_HEX}{{4}})'
      r'(?![\w:-]|\.\w)',
      re.ASCII,
    ),
    None,
  ),
  'ip': (
    re.compile(
      r'(?<![\w.-])\d{1,3}(?:\.\d{1,3}){3}(?![\w-]|\.\d)'
      rf'|(?<![\w.-])(?:{_HEX}{{0,4}}:){{2,7}}{_HEX}{{0,4}}(?![\w:-]|\.\w)',
      re.ASCII,
    ),
    _valid_ip,
  ),
  # Card networks start with 2 to 6; a timestamp in milliseconds starts with 1.
  'card': (re.compile(rf'{_NUMBER_START}[2-6]\d(?:[ -]?\d){{11,17}}{_NUMBER_END}', re.ASCII), _luhn),
  'ssn': (re.compile(r'(?<![\w-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w-])', re.ASCII), None),
  # Partita IVA: eleven digits, the last a check digit.
  'it_vat': (re.compile(rf'{_NUMBER_START}(?:IT ?)?\d{{11}}{_NUMBER_END}', re.ASCII), _luhn),
  # Runs of digits joined by single spaces, dots, dashes or parentheses, with an optional international `+`.
  'phone': (
    re.compile(rf'{_NUMBER_START}\+?\(?\d++(?:(?:\)[ .-]?|[ .-]\(?)\d++){{0,5}}{_NUMBER_END}', re.ASCII),
    _valid_phone,
  ),
  'handle': (re.compile(r'(?<![\w@.])@[A-Za-z0-9_]{2,30}\b', re.ASCII), None),
}

# The fields that hold free text. Every other field is an identifier, a name, a kind or a number, and stays as it is:
# metrics match trace, span, document and plan identifiers, so masking one would change a score.
_TEXT_FIELDS: dict[type[BaseModel], tuple[str, ...]] = {
  Sample: ('input_prompt', 'ground_truth_output'),
  Span: ('input_data', 'output_data', 'metadata', 'semantics'),
  SpanSemantics: ('request', 'answer', 'retrieval', 'planning'),
  Answer: ('text',),
  RetrievalResult: ('query', 'reason', 'items'),
  RetrievalItem: ('content', 'title', 'location', 'attributes'),
  PlanningData: ('plans', 'executed_steps'),
  PlanSnapshot: ('steps',),
  PlannedStep: ('instruction', 'parameters'),
  ExecutionStep: ('instruction', 'input', 'output', 'attributes'),
  MetricComputation: ('reasoning', 'error_message'),
}
# Ground-truth keys that identify rather than describe: claim and step IDs, plan operations and relevant IDs.
_GROUND_TRUTH_IDENTIFIERS = frozenset({'id', 'operation', 'relevant_ids'})


def _canonical(category: PiiCategory, value: str) -> str:
  """The value as its code sees it, so the same value written another way gets the same code."""
  if category in ('email', 'handle'):
    return value.lower()
  if category == 'url':
    return value
  if category == 'ip':
    return str(ipaddress.ip_address(value))
  if category in ('iban', 'it_tax_code', 'mac'):
    return re.sub(r'[\W_]', '', value).upper()
  return re.sub(r'\D', '', value)


class PiiMasker:
  """Replaces personal data with labels. Build one with `from_settings`, which returns None when masking is off."""

  def __init__(self, categories: Iterable[PiiCategory] = CATEGORIES, key: str | None = None) -> None:
    selected = set(categories)
    self._categories = [category for category in CATEGORIES if category in selected]
    self._key = None if key is None else key.encode()

  @classmethod
  def from_settings(cls, settings: PiiMaskingSettings) -> 'PiiMasker | None':
    if not settings.enabled:
      return None
    return cls(settings.categories, None if settings.key is None else settings.key.get_secret_value())

  def describe(self) -> dict[str, Any]:
    """What a run records about masking: the categories and whether labels carry codes, never the key."""
    return {'categories': list(self._categories), 'keyed': self._key is not None}

  def mask_text(self, text: str) -> str:
    # A new label can leave text that an earlier pattern now matches. Repeating until nothing changes makes masking
    # masked text a no-op.
    while True:
      masked = text
      for category in self._categories:
        pattern, valid = _RULES[category]
        masked = pattern.sub(partial(self._replace, category, pattern, valid), masked)
      if masked == text:
        return masked
      text = masked

  def mask_json(self, value: Any, keep: frozenset[str] = frozenset()) -> Any:
    """Mask every string in a JSON value; the values of keys in `keep` stay as they are."""
    if isinstance(value, str):
      return self.mask_text(value)
    if isinstance(value, dict):
      return {key: item if key in keep else self.mask_json(item, keep) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
      return [self.mask_json(item, keep) for item in value]
    return value

  def mask_model[Model: BaseModel](self, model: Model) -> Model:
    """Mask the free-text fields of a sample, a span or a metric computation."""
    return model.model_copy(update={name: self._mask_field(getattr(model, name)) for name in _TEXT_FIELDS[type(model)]})

  def mask_ground_truth(self, ground_truth: GroundTruth) -> GroundTruth:
    return ground_truth.model_copy(
      update={'ground_truth_value': self.mask_json(ground_truth.ground_truth_value, _GROUND_TRUTH_IDENTIFIERS)}
    )

  def _mask_field(self, value: Any) -> Any:
    if isinstance(value, BaseModel):
      return self.mask_model(value)
    if isinstance(value, list):
      return [self._mask_field(item) for item in value]
    return self.mask_json(value)

  def _replace(
    self,
    category: PiiCategory,
    pattern: re.Pattern[str],
    valid: Callable[[str], bool] | None,
    match: re.Match[str],
  ) -> str:
    value = match.group(0)
    if valid is None or valid(value):
      return self._label(category, value)
    # A match can run into what follows the value, such as a card's security code: mask the longest valid prefix that
    # ends before a space or a dash.
    for end in reversed([index for index, char in enumerate(value) if char in ' -']):
      prefix = value[:end]
      if pattern.fullmatch(prefix) and valid(prefix):
        return self._label(category, prefix) + value[end:]
    return value

  def _label(self, category: PiiCategory, value: str) -> str:
    if self._key is None:
      return f'[{category.upper()}]'
    code = hmac.new(self._key, f'{category}:{_canonical(category, value)}'.encode(), hashlib.sha256).hexdigest()
    return f'[{category.upper()}_{code[:8]}]'
