import json
import unittest

import httpx

from syllo_eval.evaluation.claim_extractor import ClaimExtractorMessage
from syllo_eval.infrastructure.exceptions import DataMappingError
from syllo_eval.infrastructure.orbitals import OrbitalsClaimExtractorClient
from syllo_eval.settings import OrbitalsSettings


class OrbitalsClaimExtractorClientTest(unittest.IsolatedAsyncioTestCase):
  async def test_extract_posts_expected_conversation_and_maps_claims(self) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
      requests.append(request)
      return httpx.Response(
        200,
        json={
          'extractions': {
            'claims': [
              {
                'subtype': 'Factoid',
                'content': 'Jane Doe signed for Example Corp.',
                'evidences': [],
              }
            ]
          },
          'model': 'principled-intelligence/claim-extractor-pro-2605',
          'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
          'time_taken': 0.25,
        },
      )

    config = OrbitalsSettings(
      base_url='https://orbitals.test',
      api_key='orbitals-key',
      claim_extractor_model='claim-extractor-pro-2605',
      ai_service_description='A contract assistant.',
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
      client = OrbitalsClaimExtractorClient(config, client=http_client)
      result = await client.extract(
        [
          ClaimExtractorMessage(role='user', content='Who signed the addendum?'),
          ClaimExtractorMessage(role='assistant', content='Jane Doe signed for Example Corp.'),
        ]
      )

    self.assertEqual(result.claims[0].content, 'Jane Doe signed for Example Corp.')
    self.assertEqual(result.model, 'principled-intelligence/claim-extractor-pro-2605')
    self.assertEqual(requests[0].url, 'https://orbitals.test/orbitals/claim-extractor/extract')
    self.assertEqual(requests[0].headers['X-API-Key'], 'orbitals-key')
    self.assertEqual(
      json.loads(requests[0].content),
      {
        'conversation': [
          {'role': 'user', 'content': 'Who signed the addendum?'},
          {'role': 'assistant', 'content': 'Jane Doe signed for Example Corp.'},
        ],
        'model': 'claim-extractor-pro-2605',
        'ai_service_description': 'A contract assistant.',
      },
    )

  async def test_extract_rejects_invalid_response(self) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
      del request
      return httpx.Response(200, json={'extractions': {'claims': []}})

    config = OrbitalsSettings(api_key='orbitals-key')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
      client = OrbitalsClaimExtractorClient(config, client=http_client)
      with self.assertRaises(DataMappingError):
        await client.extract([ClaimExtractorMessage(role='assistant', content='Expected answer.')])
