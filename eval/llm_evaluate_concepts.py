#!/usr/bin/env python3
"""
MapOMOP 매핑 결과 LLM 기반 정확도 평가

evaluation_blind_test_결과.xlsx의 각 unique concept에 대해 LLM으로 점수를 부여합니다.
- 0: 부정확 (entity와 concept이 관련 없거나 잘못된 매핑)
- 1: 일부정확 (부분적으로 관련 있으나 완전히 일치하지 않음)
- 2: 매우정확 (의학적으로 완전히 일치)

Usage:
    python eval/llm_evaluate_concepts.py
    python eval/llm_evaluate_concepts.py --input eval/evaluation_blind_test_결과.xlsx --output eval/llm_evaluation_result.xlsx
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

import pandas as pd

from MapOMOP.llm_client import get_llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 의료 용어 매핑 평가 전문가입니다.
OMOP CDM 표준 용어로의 매핑 정확도를 평가해 주세요.

평가 기준:
- 2 (매우정확): entity_name과 concept이 의학적으로 완전히 일치합니다. 동의어이거나 동일한 의미입니다.
- 1 (일부정확): entity_name과 concept이 부분적으로 관련되어 있으나, 완전히 일치하지는 않습니다. 상위/하위 개념 관계이거나 일부 의미가 겹칩니다.
- 0 (부정확): entity_name과 concept이 관련이 없거나 잘못된 매핑입니다.

반드시 JSON 형식으로만 응답하세요. 각 concept 문자열을 키로, 점수(0, 1, 2)를 값으로 사용합니다."""

USER_PROMPT_TEMPLATE = """다음 entity_name과 domain_id에 대해, MapOMOP 모델이 매핑한 concept들의 정확도를 평가해 주세요.

entity_name: {entity_name}
domain_id: {domain_id}

평가할 concepts:
{concepts_list}

각 concept에 대해 0, 1, 2 중 하나의 점수를 부여한 JSON 객체를 반환해 주세요.
예: {{"concept_name(id)": 2, "another_concept(id)": 1}}"""


def parse_concept_columns(df: pd.DataFrame) -> list[str]:
    """concept1, concept2, ... 컬럼명 추출"""
    return [c for c in df.columns if re.match(r"^concept\d+$", c)]


def extract_concepts_from_row(row: pd.Series, concept_cols: list[str]) -> list[str]:
    """한 행에서 비어있지 않은 concept 값들 추출 (순서 유지)"""
    concepts = []
    for col in concept_cols:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            concepts.append(str(val).strip())
    return concepts


def parse_llm_scores(response: str, concepts: list[str]) -> dict[str, int]:
    """
    LLM 응답에서 concept별 점수 파싱.
    concepts 리스트와 매칭하여 점수 반환. 매칭 실패 시 -1.
    """
    scores: dict[str, int] = {}
    try:
        # JSON 블록 추출 (```json ... ``` 또는 {...})
        text = response.strip()
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group())
        else:
            obj = json.loads(text)

        for c in concepts:
            # 정확한 키 매칭 시도
            if c in obj:
                v = obj[c]
                if isinstance(v, (int, float)):
                    scores[c] = max(0, min(2, int(v)))
                else:
                    scores[c] = -1
            else:
                # 부분 매칭 (concept 이름만)
                found = False
                for k, v in obj.items():
                    if k in c or c in k:
                        if isinstance(v, (int, float)):
                            scores[c] = max(0, min(2, int(v)))
                        else:
                            scores[c] = -1
                        found = True
                        break
                if not found:
                    scores[c] = -1
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"LLM 응답 파싱 실패: {e}")
        for c in concepts:
            scores[c] = -1
    return scores


def evaluate_row(
    llm_client,
    entity_name: str,
    domain_id: str,
    concepts: list[str],
) -> dict[str, int]:
    """한 entity의 concept들에 대해 LLM 평가 수행"""
    if not concepts:
        return {}

    concepts_list = "\n".join(f"- {c}" for c in concepts)
    prompt = USER_PROMPT_TEMPLATE.format(
        entity_name=entity_name,
        domain_id=domain_id,
        concepts_list=concepts_list,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = llm_client.chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
        json_mode=True,
    )
    if not response:
        return {c: -1 for c in concepts}
    return parse_llm_scores(response, concepts)


def run_evaluation(
    input_path: str,
    output_path: str,
    limit: int | None = None,
) -> None:
    """메인 평가 실행"""
    df = pd.read_excel(input_path)
    concept_cols = parse_concept_columns(df)
    if not concept_cols:
        raise ValueError("concept1, concept2, ... 컬럼을 찾을 수 없습니다.")

    llm_client = get_llm_client()
    if not llm_client.is_initialized:
        raise RuntimeError("LLM 클라이언트 초기화 실패. OPENAI_API_KEY 또는 .env 설정을 확인하세요.")

    rows = df.to_dict("records")
    if limit:
        rows = rows[:limit]
        logger.info(f"처리 제한: 상위 {limit}건만 평가")

    results = []
    total_concepts = 0
    score_counts = {0: 0, 1: 0, 2: 0, -1: 0}

    for i, row in enumerate(rows):
        entity_name = str(row.get("entity_name", "")).strip()
        domain_id = str(row.get("domain_id", "")).strip()
        concepts = extract_concepts_from_row(row, concept_cols)

        if not concepts:
            new_row = dict(row)
            for j in range(len(concept_cols)):
                new_row[f"llm_score{j + 1}"] = "N/A"
            results.append(new_row)
            continue

        scores = evaluate_row(llm_client, entity_name, domain_id, concepts)
        new_row = dict(row)
        for j, col in enumerate(concept_cols):
            if j < len(concepts):
                s = scores.get(concepts[j], -1)
                new_row[f"llm_score{j + 1}"] = s if s >= 0 else "ERR"
                if s >= 0:
                    score_counts[s] = score_counts.get(s, 0) + 1
                    total_concepts += 1
                else:
                    score_counts[-1] = score_counts.get(-1, 0) + 1
            else:
                new_row[f"llm_score{j + 1}"] = "N/A"
        results.append(new_row)

        if (i + 1) % 50 == 0:
            logger.info(f"진행: {i + 1}/{len(rows)}")

    out_df = pd.DataFrame(results)

    # 엑셀 저장
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="LLM평가")

    # 정확도 지표 계산
    valid = total_concepts - score_counts.get(-1, 0)
    if valid > 0:
        acc_2 = score_counts.get(2, 0) / valid * 100
        acc_1_or_2 = (score_counts.get(1, 0) + score_counts.get(2, 0)) / valid * 100
        weighted = (score_counts.get(2, 0) * 2 + score_counts.get(1, 0) * 1) / (valid * 2) * 100
    else:
        acc_2 = acc_1_or_2 = weighted = 0.0

    logger.info("=" * 60)
    logger.info("LLM 평가 완료")
    logger.info("=" * 60)
    logger.info(f"총 평가 concept 수: {total_concepts}")
    logger.info(f"  - 매우정확(2): {score_counts.get(2, 0)}")
    logger.info(f"  - 일부정확(1): {score_counts.get(1, 0)}")
    logger.info(f"  - 부정확(0):   {score_counts.get(0, 0)}")
    logger.info(f"  - 파싱실패(-1): {score_counts.get(-1, 0)}")
    logger.info("-" * 60)
    logger.info(f"매우정확 비율 (2점만): {acc_2:.1f}%")
    logger.info(f"일부정확 이상 비율 (1+2점): {acc_1_or_2:.1f}%")
    logger.info(f"가중 평균 정확도 (2=100%, 1=50%, 0=0%): {weighted:.1f}%")
    logger.info("=" * 60)
    logger.info(f"결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MapOMOP 매핑 결과 LLM 기반 정확도 평가"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(Path(__file__).parent / "evaluation_blind_test_결과.xlsx"),
        help="입력 엑셀 경로 (기본: eval/evaluation_blind_test_결과.xlsx)",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).parent / "llm_evaluation_result.xlsx"),
        help="출력 엑셀 경로 (기본: eval/llm_evaluation_result.xlsx)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="평가할 행 수 제한 (테스트용)",
    )
    args = parser.parse_args()

    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
