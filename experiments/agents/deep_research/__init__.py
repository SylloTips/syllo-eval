"""The deep_research example of LangChain's deepagents, at the commit of its 0.7.34 release, under its MIT license.

``prompts.py`` is ``examples/deep_research/research_agent/prompts.py`` byte for byte. ``tools.py`` keeps the example's
``think_tool`` and drops its web search, which the knowledge-base search tool replaces. ``agents/deep_agent.py`` wires
them as the example's ``agent.py`` does, and edits the prompts for the knowledge base.
"""

REPOSITORY = 'https://github.com/langchain-ai/deepagents'
COMMIT = '9ea3f1912abd8b7eecb504d78ef99fcf279bf561'
