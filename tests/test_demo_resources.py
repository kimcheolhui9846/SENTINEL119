import unittest
from unittest.mock import AsyncMock, Mock, patch

from backend import dispatch_map, transfer


class VehicleAssignmentTest(unittest.IsolatedAsyncioTestCase):
    async def test_equal_eta_is_stable_across_permutations_and_shortlist(self):
        point = {"lat": 37.5, "lon": 127.0}
        units = [{"id": key, "kind": "ambulance", **point} for key in "ABCDE"]
        steps = [{"kind": "ambulance"}]
        for lineup in (units, units[::-1]):
            with patch.object(dispatch_map, "fleet", return_value=lineup), patch.object(
                dispatch_map, "route", AsyncMock(return_value={"duration_s": 12})
            ):
                assigned = await dispatch_map.assign(point, steps)
                self.assertEqual(assigned[0]["unit"]["id"], "A")
                assigned = await dispatch_map.assign(point, steps, busy=["A"])
                self.assertEqual(assigned[0]["unit"]["id"], "B")

    async def test_shorter_actual_eta_wins(self):
        point = {"lat": 37.5, "lon": 127.0}
        units = [{"id": key, "kind": "ambulance", **point} for key in "AB"]
        async def route(unit, incident):
            return {"duration_s": 12 if unit["id"] == "A" else 10}
        with patch.object(dispatch_map, "fleet", return_value=units), patch.object(dispatch_map, "route", route):
            assigned = await dispatch_map.assign(point, [{"kind": "ambulance"}])
            self.assertEqual(assigned[0]["unit"]["id"], "B")


class HospitalChoiceTest(unittest.IsolatedAsyncioTestCase):
    hospitals = [{"hpid": "A", "name": "병원 가"}, {"hpid": "B", "name": "병원 나"}]
    rule = transfer.transfer_rule()
    turns = [{"speaker": 1, "text": "가상 신고입니다."}]

    async def score(self, scores, *, reasoned=False, finished=True, omit=()):
        response = {"results": {"probabilities": scores}, "direct": {"A": .8, "B": .2},
                    "reasoned": scores, "reasoning": "synthetic", "reasoning_finished": finished,
                    "reasoning_tokens": 4, "reasoning_ms": 10}
        for field in omit:
            response.pop(field)
        client = AsyncMock()
        client.__aenter__.return_value.post.return_value = Mock(json=Mock(return_value=response))
        with patch.dict("os.environ", {"SENTINEL_ZORO_URL": "http://fake.invalid"}), patch.object(
            transfer.httpx, "AsyncClient", return_value=client
        ), patch.object(transfer, "_criterion", return_value="sample"):
            if reasoned:
                return await transfer.zoro_reason(self.turns, self.rule, self.hospitals, {})
            return await transfer.zoro_choice(self.turns, self.rule, self.hospitals)

    async def test_invalid_distribution_or_tie_never_selects_a_hospital(self):
        for scores in ({"A": .5, "B": .5}, {"A": .8, "B": .2, "X": 0}, {"A": 1},
                       {"A": float("nan"), "B": .2}, {"A": 1.3, "B": -.3}, {"A": .1, "B": .2},
                       {"A": True, "B": 0}, {"A": ".8", "B": .2}):
            for reasoned in (False, True):
                with self.subTest(scores=scores, reasoned=reasoned):
                    self.assertIsNone(await self.score(scores, reasoned=reasoned))

    async def test_valid_choice_and_incomplete_reasoning(self):
        for reasoned in (True, False):
            choice = await self.score({"A": .2, "B": .8}, reasoned=reasoned)
            self.assertEqual(choice["choice"], "B")
        self.assertIsNone(await self.score({"A": .2, "B": .8}, reasoned=True, finished=False))
        self.assertIsNone(await self.score({"A": .2, "B": .8}, reasoned=True, omit=("reasoning",)))

    async def test_explanation_runs_once_and_preserves_optional_reasoning(self):
        for requested in (False, True):
            body = transfer.TransferRequest(lat=37.5, lon=127, turns=self.turns, explain=True, reason=requested)
            reasoning = {"choice": "B"}
            with patch.object(transfer, "candidates", AsyncMock(return_value=("live", []))), patch.object(
                transfer, "zoro_choice", AsyncMock(return_value={"choice": "A", "question": {}})
            ), patch.object(transfer, "explain", AsyncMock(return_value={"summary": []})) as explain, patch.object(
                transfer, "zoro_reason", AsyncMock(return_value=reasoning)
            ) as optional:
                result = await transfer.transfer(body)
                self.assertEqual(result["reason"]["reasoned"], reasoning if requested else None)
                explain.assert_awaited_once()
                self.assertEqual(optional.await_count, int(requested))
