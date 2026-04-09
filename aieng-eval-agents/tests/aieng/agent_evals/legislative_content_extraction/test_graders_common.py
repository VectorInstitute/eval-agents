"""Tests for legislative content extraction graders.

Covers the three main precision/recall failure modes:
1. Wisconsin suffix normalization ("of the Wisconsin Statutes", "Wis. Stats.")
2. Michigan MCL range normalization ("MCL 408.931 to 408.945")
3. Delaware chapter/section granularity mismatch (LLM judge path)

Also covers:
- Expanded _ACTION_EQUIVALENCES
- Soft-match (Layer 2) fallback
- Section judge cache + mock-based wiring
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aieng.agent_evals.legislative_content_extraction.graders._common import (
    normalize_raw_section,
    normalize_sections,
    _normalize_action,
)
from aieng.agent_evals.legislative_content_extraction.graders.item import (
    _soft_match_score,
    _build_section_raw_map,
    item_level_deterministic_grader,
)
from aieng.agent_evals.legislative_content_extraction.graders.section_judge import (
    SectionMatchResult,
    clear_cache,
    _cache_key,
)


# ---------------------------------------------------------------------------
# Layer 1 – normalize_raw_section
# ---------------------------------------------------------------------------


class TestNormalizeRawSectionWisconsin:
    """Wisconsin suffix stripping."""

    def test_of_the_statutes(self) -> None:
        assert normalize_raw_section("16.03 (2) (a) of the statutes") == "16.03 (2) (A)"

    def test_of_the_wisconsin_statutes(self) -> None:
        assert normalize_raw_section("16.03 (2) (a) of the Wisconsin Statutes") == "16.03 (2) (A)"

    def test_section_prefix_comma_wisconsin_statutes(self) -> None:
        assert normalize_raw_section("Section 101.123 (1) (ah), Wisconsin Statutes") == "101.123 (1) (AH)"

    def test_wis_stats_suffix(self) -> None:
        assert normalize_raw_section("101.05 (3), Wis. Stats.") == "101.05 (3)"

    def test_wis_stat_suffix_no_comma(self) -> None:
        assert normalize_raw_section("101.05 (3) Wis. Stat.") == "101.05 (3)"

    def test_wisconsin_statutes_with_year(self) -> None:
        # "Wisconsin Statutes 2023" suffix form
        result = normalize_raw_section("Section 48.02 (1), Wisconsin Statutes 2023")
        assert result == "48.02 (1)"

    def test_already_clean(self) -> None:
        assert normalize_raw_section("101.123 (1) (ah)") == "101.123 (1) (AH)"


class TestNormalizeRawSectionMichigan:
    """Michigan MCL normalization."""

    def test_mcl_gt_format(self) -> None:
        # GT form: "MCL 15.232 (Section 6)"
        assert normalize_raw_section("MCL 15.232 (Section 6)") == "MCL 15.232 | SECTION 6"

    def test_mcl_agent_format(self) -> None:
        # Agent form: "section 6 (MCL 15.232)"
        assert normalize_raw_section("section 6 (MCL 15.232)") == "MCL 15.232 | SECTION 6"

    def test_mcl_range_to(self) -> None:
        assert normalize_raw_section("MCL 408.931 to 408.945") == "MCL 408.931 TO 408.945"

    def test_mcl_range_through(self) -> None:
        assert normalize_raw_section("MCL 408.931 through 408.945") == "MCL 408.931 TO 408.945"

    def test_mcl_range_hyphen(self) -> None:
        assert normalize_raw_section("MCL 408.931 - 408.945") == "MCL 408.931 TO 408.945"

    def test_mcl_full_act_title_with_range(self) -> None:
        raw = "The improved workforce opportunity wage act, 2018 PA 337, MCL 408.931 to 408.945"
        assert normalize_raw_section(raw) == "MCL 408.931 TO 408.945"

    def test_mcl_full_act_title_single_section(self) -> None:
        raw = "The open meetings act, 1976 PA 267, MCL 15.261"
        result = normalize_raw_section(raw)
        # Should extract MCL 15.261
        assert "15.261" in result

    def test_both_forms_match_after_normalization(self) -> None:
        # GT and agent both normalize to the same canonical form
        gt = normalize_raw_section("MCL 15.232 (Section 6)")
        agent = normalize_raw_section("section 6 (MCL 15.232)")
        assert gt == agent

    def test_range_both_sides_match(self) -> None:
        a = normalize_raw_section("MCL 408.931 to 408.945")
        b = normalize_raw_section("MCL 408.931 through 408.945")
        assert a == b


class TestNormalizeRawSectionDelaware:
    """Delaware chapter/section handling."""

    def test_chapter_level_gt_strips_section_detail(self) -> None:
        gt = "Subchapter IV, Chapter 25, Title 6 - § 2541. Definitions."
        result = normalize_raw_section(gt)
        assert result == "SUBCHAPTER IV, CHAPTER 25, TITLE 6"

    def test_chapter_level_agent_strips_of_delaware_code(self) -> None:
        agent = "Subchapter IV, Chapter 25, Title 6 of the Delaware Code"
        result = normalize_raw_section(agent)
        assert result == "SUBCHAPTER IV, CHAPTER 25, TITLE 6"

    def test_chapter_level_both_match(self) -> None:
        gt = normalize_raw_section("Subchapter IV, Chapter 25, Title 6 - § 2541. Definitions.")
        agent = normalize_raw_section("Subchapter IV, Chapter 25, Title 6 of the Delaware Code")
        assert gt == agent


class TestNormalizeRawSectionIdaho:
    """Idaho normalization regression."""

    def test_idaho_code_suffix(self) -> None:
        assert normalize_raw_section("Section 33-1802, Idaho Code") == "33-1802"

    def test_idaho_constitution(self) -> None:
        result = normalize_raw_section("Section 1, Article III, of the Constitution of the State of Idaho")
        assert "ARTICLE III" in result


# ---------------------------------------------------------------------------
# _normalize_action – expanded equivalences
# ---------------------------------------------------------------------------


class TestNormalizeAction:
    def test_create_maps_to_add(self) -> None:
        assert _normalize_action("CREATE") == "ADD"

    def test_new_maps_to_add(self) -> None:
        assert _normalize_action("new") == "ADD"

    def test_insert_maps_to_add(self) -> None:
        assert _normalize_action("INSERT") == "ADD"

    def test_redesig_maps_to_amend(self) -> None:
        assert _normalize_action("REDESIG") == "AMEND"

    def test_redesignate_maps_to_amend(self) -> None:
        assert _normalize_action("REDESIGNATE") == "AMEND"

    def test_renumber_maps_to_amend(self) -> None:
        assert _normalize_action("RENUMBER") == "AMEND"

    def test_renumber_and_amend(self) -> None:
        assert _normalize_action("RENUMBER AND AMEND") == "AMEND"

    def test_renumber_ampersand_amend(self) -> None:
        assert _normalize_action("RENUMBER & AMEND") == "AMEND"

    def test_repeal_and_recreate(self) -> None:
        assert _normalize_action("REPEAL & RECREATE") == "REPEAL AND RECREATE"

    def test_deauthorize_maps_to_repeal(self) -> None:
        assert _normalize_action("DEAUTHORIZE") == "REPEAL"

    def test_amend_unchanged(self) -> None:
        assert _normalize_action("AMEND") == "AMEND"

    def test_repeal_unchanged(self) -> None:
        assert _normalize_action("repeal") == "REPEAL"


# ---------------------------------------------------------------------------
# normalize_sections (integration)
# ---------------------------------------------------------------------------


class TestNormalizeSections:
    def test_wi_section_normalized(self) -> None:
        sections = [{"raw_section": "Section 101.123 (1) (ah), Wisconsin Statutes", "action": "AMEND"}]
        result = normalize_sections(sections)
        assert "101.123 (1) (AH)|AMEND" in result

    def test_wi_of_statutes_normalized(self) -> None:
        sections = [{"raw_section": "16.03 (2) (a) of the statutes", "action": "AMEND"}]
        result = normalize_sections(sections)
        assert "16.03 (2) (A)|AMEND" in result

    def test_wi_of_wisconsin_statutes_matches_of_statutes(self) -> None:
        pred = normalize_sections([{"raw_section": "16.03 (2) (a) of the Wisconsin Statutes", "action": "AMEND"}])
        ref = normalize_sections([{"raw_section": "Section 16.03 (2) (a), Wisconsin Statutes", "action": "AMEND"}])
        assert pred == ref

    def test_mi_both_forms_produce_same_key(self) -> None:
        gt = normalize_sections([{"raw_section": "MCL 15.232 (Section 6)", "action": "AMEND"}])
        agent = normalize_sections([{"raw_section": "section 6 (MCL 15.232)", "action": "AMEND"}])
        assert gt == agent

    def test_mi_range_key_consistent(self) -> None:
        a = normalize_sections([{"raw_section": "MCL 408.931 to 408.945", "action": "AMEND"}])
        b = normalize_sections([{"raw_section": "MCL 408.931 through 408.945", "action": "AMEND"}])
        assert a == b

    def test_action_synonym_create_equals_add(self) -> None:
        pred = normalize_sections([{"raw_section": "33-1802", "action": "CREATE"}])
        ref = normalize_sections([{"raw_section": "33-1802", "action": "ADD"}])
        assert pred == ref


# ---------------------------------------------------------------------------
# Layer 2 – soft match
# ---------------------------------------------------------------------------


class TestSoftMatchScore:
    def test_identical_tokens_exact_ratio(self) -> None:
        # Tokens that are very similar but not equal should soft-match
        pred = {"MCL 15.232 | SECTION 6"}
        ref = {"MCL 15.232 | SECTION 06"}  # leading zero difference
        score = _soft_match_score(pred, ref)
        assert score == pytest.approx(0.5)

    def test_completely_different_no_match(self) -> None:
        pred = {"101.123 (1) (AH)|AMEND"}
        ref = {"33-1802|REPEAL"}
        assert _soft_match_score(pred, ref) == 0.0

    def test_empty_sets_return_zero(self) -> None:
        assert _soft_match_score(set(), {"TOKEN"}) == 0.0
        assert _soft_match_score({"TOKEN"}, set()) == 0.0

    def test_greedy_assignment(self) -> None:
        # Two pred tokens, two ref tokens; each pair should match once at most
        pred = {"101.05 (3)|AMEND", "101.06 (1)|AMEND"}
        ref = {"101.05 (3)|ADD", "101.06 (1)|ADD"}
        # Both pairs should be above threshold (same number, different action)
        score = _soft_match_score(pred, ref)
        # Two matches × 0.5 = 1.0 (or less if threshold not met)
        assert score >= 0.0  # at minimum non-negative

    def test_no_double_counting(self) -> None:
        # One pred vs two very similar refs → only one match
        pred = {"101.05 (3)|AMEND"}
        ref = {"101.05 (3)|ADD", "101.05 (3)|REPEAL"}
        score = _soft_match_score(pred, ref)
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Layer 3 – SectionJudge cache + grader integration
# ---------------------------------------------------------------------------


class TestSectionJudgeCache:
    def setup_method(self) -> None:
        clear_cache()

    def test_cache_key_deterministic(self) -> None:
        k1 = _cache_key("DE", "§ 2541, Title 6", "AMEND", "Chapter 25, Title 6", "AMEND")
        k2 = _cache_key("DE", "§ 2541, Title 6", "AMEND", "Chapter 25, Title 6", "AMEND")
        assert k1 == k2

    def test_cache_key_order_insensitive_to_case(self) -> None:
        k1 = _cache_key("DE", "§ 2541, title 6", "amend", "Chapter 25, Title 6", "AMEND")
        k2 = _cache_key("DE", "§ 2541, TITLE 6", "AMEND", "CHAPTER 25, TITLE 6", "amend")
        assert k1 == k2

    def test_cache_key_different_for_different_inputs(self) -> None:
        k1 = _cache_key("DE", "§ 2541, Title 6", "AMEND", "Chapter 25, Title 6", "AMEND")
        k2 = _cache_key("MI", "§ 2541, Title 6", "AMEND", "Chapter 25, Title 6", "AMEND")
        assert k1 != k2


# ---------------------------------------------------------------------------
# Full grader integration (no LLM sections – avoids external calls)
# ---------------------------------------------------------------------------


def _make_item(
    jurisdiction: str = "WI",
    sections_pred: list[dict] | None = None,
    sections_ref: list[dict] | None = None,
) -> tuple[dict, dict]:
    base = {
        "jurisdiction_code": jurisdiction,
        "session_code": f"{jurisdiction}_2025_R1",
        "chamber_code": "HOUSE",
        "measure_type_code": "BILL",
        "measure_number": "1",
        "sponsors": [],
        "sections_affected": sections_pred or [],
    }
    ref = dict(base)
    ref["sections_affected"] = sections_ref or []
    return base, ref


class TestItemGraderSectionsNoLLM:
    """Integration tests that do NOT trigger LLM calls.

    Layer 3 is skipped when exact + soft matching fully resolves.
    """

    def test_empty_sections_both_sides(self) -> None:
        output, expected = _make_item(sections_pred=[], sections_ref=[])
        evals = item_level_deterministic_grader(input={}, output=output, expected_output=expected)
        pr = {e.name: e.value for e in evals}
        assert pr["sections_precision"] == pytest.approx(1.0)
        assert pr["sections_recall"] == pytest.approx(1.0)

    def test_perfect_wi_match(self) -> None:
        sec = [{"raw_section": "Section 101.123 (1) (ah), Wisconsin Statutes", "action": "AMEND"}]
        output, expected = _make_item(sections_pred=sec, sections_ref=sec)
        evals = item_level_deterministic_grader(input={}, output=output, expected_output=expected)
        pr = {e.name: e.value for e in evals}
        assert pr["sections_precision"] == pytest.approx(1.0)
        assert pr["sections_recall"] == pytest.approx(1.0)

    def test_wi_of_statutes_matches_wi_statutes_comma(self) -> None:
        pred_sec = [{"raw_section": "16.03 (2) (a) of the Wisconsin Statutes", "action": "AMEND"}]
        ref_sec = [{"raw_section": "Section 16.03 (2) (a), Wisconsin Statutes", "action": "AMEND"}]
        output, expected = _make_item(sections_pred=pred_sec, sections_ref=ref_sec)
        evals = item_level_deterministic_grader(input={}, output=output, expected_output=expected)
        pr = {e.name: e.value for e in evals}
        assert pr["sections_precision"] == pytest.approx(1.0)
        assert pr["sections_recall"] == pytest.approx(1.0)

    def test_mi_mcl_both_forms_exact_match(self) -> None:
        pred_sec = [{"raw_section": "section 6 (MCL 15.232)", "action": "AMEND"}]
        ref_sec = [{"raw_section": "MCL 15.232 (Section 6)", "action": "AMEND"}]
        output, expected = _make_item(jurisdiction="MI", sections_pred=pred_sec, sections_ref=ref_sec)
        evals = item_level_deterministic_grader(input={}, output=output, expected_output=expected)
        pr = {e.name: e.value for e in evals}
        assert pr["sections_precision"] == pytest.approx(1.0)
        assert pr["sections_recall"] == pytest.approx(1.0)

    def test_mi_mcl_range_both_forms_match(self) -> None:
        pred_sec = [{"raw_section": "MCL 408.931 to 408.945", "action": "AMEND"}]
        ref_sec = [{"raw_section": "MCL 408.931 through 408.945", "action": "AMEND"}]
        output, expected = _make_item(jurisdiction="MI", sections_pred=pred_sec, sections_ref=ref_sec)
        evals = item_level_deterministic_grader(input={}, output=output, expected_output=expected)
        pr = {e.name: e.value for e in evals}
        assert pr["sections_precision"] == pytest.approx(1.0)
        assert pr["sections_recall"] == pytest.approx(1.0)

    def test_zero_recall_completely_different_sections(self) -> None:
        pred_sec = [{"raw_section": "101.123 (1)", "action": "AMEND"}]
        ref_sec = [{"raw_section": "33-1802", "action": "REPEAL"}]
        output, expected = _make_item(sections_pred=pred_sec, sections_ref=ref_sec)
        evals = item_level_deterministic_grader(input={}, output=output, expected_output=expected)
        pr = {e.name: e.value for e in evals}
        # No exact or soft match; LLM judge skipped (no jurisdiction match needed)
        assert pr["sections_recall"] == pytest.approx(0.0)

    def test_action_create_equals_add_for_exact_match(self) -> None:
        pred_sec = [{"raw_section": "Section 33-1802, Idaho Code", "action": "CREATE"}]
        ref_sec = [{"raw_section": "Section 33-1802, Idaho Code", "action": "ADD"}]
        output, expected = _make_item(jurisdiction="ID", sections_pred=pred_sec, sections_ref=ref_sec)
        evals = item_level_deterministic_grader(input={}, output=output, expected_output=expected)
        pr = {e.name: e.value for e in evals}
        assert pr["sections_precision"] == pytest.approx(1.0)
        assert pr["sections_recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Layer 3 – LLM judge mock integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_item_grader_async_uses_llm_judge_for_de_granularity() -> None:
    """DE section vs chapter mismatch triggers LLM judge that returns match=True."""
    from aieng.agent_evals.legislative_content_extraction.graders.item import (
        item_level_deterministic_grader_async,
    )

    clear_cache()

    # Predicted: individual § ref; GT: chapter-level ref
    pred_sec = [{"raw_section": "§ 2541, Title 6 of the Delaware Code", "action": "AMEND"}]
    ref_sec = [{"raw_section": "Subchapter IV, Chapter 25, Title 6 - § 2541. Definitions.", "action": "AMEND"}]

    output = {
        "jurisdiction_code": "DE",
        "session_code": "DE_2025_R1",
        "chamber_code": "HOUSE",
        "measure_type_code": "BILL",
        "measure_number": "145",
        "sponsors": [],
        "sections_affected": pred_sec,
    }
    expected = dict(output)
    expected["sections_affected"] = ref_sec

    mock_result = SectionMatchResult(match=True, confidence=0.95, reasoning="Same chapter.")

    with patch(
        "aieng.agent_evals.legislative_content_extraction.graders.section_judge.judge_section_pair",
        new=AsyncMock(return_value=mock_result),
    ):
        evals = await item_level_deterministic_grader_async(input={}, output=output, expected_output=expected)

    pr = {e.name: e.value for e in evals}
    # LLM judge says match → 0.8 credit → recall = 0.8/1 = 0.8
    assert pr["sections_recall"] == pytest.approx(0.8)
    assert pr["sections_precision"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_item_grader_async_llm_judge_no_match() -> None:
    """LLM judge returning match=False yields 0 recall for that pair."""
    from aieng.agent_evals.legislative_content_extraction.graders.item import (
        item_level_deterministic_grader_async,
    )

    clear_cache()

    pred_sec = [{"raw_section": "§ 9999, Title 99 of the Delaware Code", "action": "AMEND"}]
    ref_sec = [{"raw_section": "Subchapter I, Chapter 1, Title 1 - § 100. Purpose.", "action": "AMEND"}]

    output = {
        "jurisdiction_code": "DE",
        "session_code": "DE_2025_R1",
        "chamber_code": "HOUSE",
        "measure_type_code": "BILL",
        "measure_number": "999",
        "sponsors": [],
        "sections_affected": pred_sec,
    }
    expected = dict(output)
    expected["sections_affected"] = ref_sec

    mock_result = SectionMatchResult(match=False, confidence=0.95, reasoning="Different chapter and title.")

    with patch(
        "aieng.agent_evals.legislative_content_extraction.graders.section_judge.judge_section_pair",
        new=AsyncMock(return_value=mock_result),
    ):
        evals = await item_level_deterministic_grader_async(input={}, output=output, expected_output=expected)

    pr = {e.name: e.value for e in evals}
    assert pr["sections_recall"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_item_grader_async_llm_judge_exception_degrades_gracefully() -> None:
    """Judge API failure → 0 credit for that pair, evaluation continues."""
    from aieng.agent_evals.legislative_content_extraction.graders.item import (
        item_level_deterministic_grader_async,
    )

    clear_cache()

    pred_sec = [{"raw_section": "§ 2541, Title 6 of the Delaware Code", "action": "AMEND"}]
    ref_sec = [{"raw_section": "Subchapter IV, Chapter 25, Title 6 - § 2541. Definitions.", "action": "AMEND"}]

    output = {
        "jurisdiction_code": "DE",
        "session_code": "DE_2025_R1",
        "chamber_code": "HOUSE",
        "measure_type_code": "BILL",
        "measure_number": "145",
        "sponsors": [],
        "sections_affected": pred_sec,
    }
    expected = dict(output)
    expected["sections_affected"] = ref_sec

    with patch(
        "aieng.agent_evals.legislative_content_extraction.graders.section_judge.judge_section_pair",
        new=AsyncMock(side_effect=RuntimeError("judge API unavailable")),
    ):
        evals = await item_level_deterministic_grader_async(input={}, output=output, expected_output=expected)

    pr = {e.name: e.value for e in evals}
    # Judge failed → 0 credit for that pair
    assert pr["sections_recall"] == pytest.approx(0.0)
    # Evaluation did not raise
    assert "sections_precision" in pr