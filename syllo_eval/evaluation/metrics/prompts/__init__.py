from functools import cache
from importlib.resources import files
from string import Template


@cache
def _paragraphs(path: str) -> tuple[Template, ...]:
  text = files(__name__).joinpath(path).read_text(encoding='utf-8')
  return tuple(Template(paragraph) for paragraph in text.strip('\n').split('\n\n'))


def render_prompt(path: str, **values: str | None) -> str:
  """Render a packaged prompt template, omitting every paragraph that uses a value that is None."""
  return '\n\n'.join(
    paragraph.substitute(values)
    for paragraph in _paragraphs(path)
    if all(values[name] is not None for name in paragraph.get_identifiers())
  )
