#!/usr/bin/env python3
"""
로그 파일을 파싱하여 mapping_common.save_xlsx 형식의 Excel로 변환.

매핑 실행 중(중간)에도 로그만 있으면 Excel로 변환 가능.
CSV와 병합하여 ground_truth, id 등 보완.

- 맨 앞 시트: 현황 (All Same, Correct, Mapped Concept 1~N)
- 이후 시트: Run 1 상세, Run 2 상세, ... Run N 상세
- Stage 1/2/3: 로그 원문 그대로 저장
- Best Concept ID: Stage 3의 최고점 후보에서 파싱

Usage:
    python scripts/log_to_xlsx.py test_logs/mapping_snuh_20260303_115235.log
    python scripts/log_to_xlsx.py test_logs/mapping_snuh_20260303_115235.log --csv data/snuh-baseline-mapping-data.csv
    python scripts/log_to_xlsx.py test_logs/mapping_snuh_20260303_115235.log --watch  # 10초마다 갱신
"""

import argparse
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pandas as pd

try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from mapping_common import DATA_SOURCES, XLSX_HEADERS, SUMMARY_HEADERS


# -----------------------------------------------------------------------------
# 로그 파싱 정규식
# -----------------------------------------------------------------------------
RE_START = re.compile(
    r"Starting single-domain mapping: (.+)$",
    re.MULTILINE,
)
RE_TARGET_DOMAIN = re.compile(
    r"Target domain: (\w+)$",
    re.MULTILINE,
)
# Stage 3: concept_name (concept_id) score - Best Concept ID 추출용
RE_STAGE3_SCORE = re.compile(
    r"^\s*(.+?)\s+\((\d+)\)\s+([\d.]+)\s*$",
)
# #N Entity: 정답/오답/실패
RE_STATUS = re.compile(
    r"#(\d+)\s+(.+?):\s+(정답|오답|실패)$",
)
# 매핑 Run N/M
RE_RUN_BOUNDARY = re.compile(
    r"매핑 Run (\d+)/(\d+)",
)


def _extract_log_message(line: str) -> str | None:
    """로그 라인에서 메시지 부분 추출. ' - INFO - ' 이후."""
    if " - INFO - " in line:
        return line.split(" - INFO - ", 1)[1].strip()
    return None


def parse_log_blocks(log_path: Path) -> tuple[list[dict], list[tuple[int, int, int]], int, set[int]]:
    """
    로그 파일을 파싱하여 엔티티별 블록 리스트 반환.
    각 블록: entity_name, domain, stage1/2/3_raw_text (로그 원문), best_* (Stage3 최고점에서 파싱), test_index, mapping_correct
    Returns: (blocks, run_boundaries, num_runs, completed_run_nums)
    completed_run_nums: 마무리된 run 번호 집합 (진행 중인 run 제외)
    """
    text = log_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Run 경계 감지: "매핑 Run N/M"
    run_boundaries: list[tuple[int, int, int]] = []  # (line_idx, run_num, total_runs)
    for i, line in enumerate(lines):
        msg = _extract_log_message(line)
        if msg:
            m = RE_RUN_BOUNDARY.search(msg)
            if m:
                run_num, total = int(m.group(1)), int(m.group(2))
                run_boundaries.append((i, run_num, total))
    num_runs = run_boundaries[-1][2] if run_boundaries else 1
    # 마무리된 run만: 다음 run 경계가 있으면 해당 run은 완료. 마지막 run은 진행 중.
    # 단일 run(num_runs=1)이면 해당 run 포함. 2개 이상이면 마지막 제외.
    if len(run_boundaries) > 1:
        completed_run_nums = {rn for (_, rn, _) in run_boundaries[:-1]}
    elif run_boundaries:
        completed_run_nums = {run_boundaries[0][1]}  # 단일 run
    else:
        completed_run_nums = {1}  # run 마커 없으면 run 1 포함

    # "Starting single-domain mapping:" 기준으로 블록 분할
    block_starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        msg = _extract_log_message(line)
        if not msg:
            continue
        m = RE_START.search(msg)
        if m:
            entity_name = m.group(1).strip()
            entity_key = " ".join(entity_name.split())
            block_starts.append((i, entity_key))

    def _run_at_line(line_idx: int) -> int:
        """line_idx가 속한 run 번호 (1-based)."""
        run = 1
        for (bound_idx, rn, _) in run_boundaries:
            if line_idx >= bound_idx:
                run = rn
        return run

    # #N Entity: status 라인을 로그 순서대로 수집, 엔티티별 큐로 구성
    status_by_entity: dict[str, list[tuple[int, str, bool]]] = {}
    for line in lines:
        msg = _extract_log_message(line)
        if msg:
            m = RE_STATUS.search(msg)
            if m:
                test_index, entity_display, mapping_correct = int(m.group(1)), m.group(2).strip(), m.group(3) == "정답"
                entity_key = " ".join(entity_display.split())
                if entity_key not in status_by_entity:
                    status_by_entity[entity_key] = []
                status_by_entity[entity_key].append((test_index, entity_display, mapping_correct))

    blocks: list[dict] = []
    for block_idx, (start_idx, entity_key) in enumerate(block_starts):
        end_idx = block_starts[block_idx + 1][0] if block_idx + 1 < len(block_starts) else len(lines)
        block_lines = [lines[i] for i in range(start_idx, end_idx)]

        entity_name = entity_key
        domain = "All"
        stage1_raw: list[str] = []
        stage2_raw: list[str] = []
        stage3_raw: list[str] = []
        best_domain = None
        best_concept_id = None
        best_concept_name = None
        best_score = 0.0
        in_stage1 = False
        in_stage2 = False
        in_stage3 = False

        for line in block_lines:
            msg = _extract_log_message(line)
            if not msg:
                continue

            if "Target domain:" in msg:
                dm = RE_TARGET_DOMAIN.search(msg)
                if dm:
                    domain = dm.group(1)

            if "Stage 1: Candidate Retrieval" in msg:
                in_stage1 = True
                in_stage2 = False
                in_stage3 = False
                continue
            if "Stage 2: Standard Concept Collection" in msg:
                in_stage1 = False
                in_stage2 = True
                in_stage3 = False
                continue
            if "Stage 3: Scoring Results" in msg:
                in_stage1 = False
                in_stage2 = False
                in_stage3 = True
                continue

            if in_stage1:
                stage1_raw.append(msg)
            elif in_stage2:
                stage2_raw.append(msg)
            elif in_stage3:
                stage3_raw.append(msg)

        # Stage 3에서 Best Concept ID 파싱: "concept_name (concept_id) score" 형식 중 최고점
        stage3_scores: list[tuple[str, int, float]] = []
        for msg in stage3_raw:
            if msg.strip().startswith("→"):
                continue
            m = RE_STAGE3_SCORE.search(msg)
            if m:
                cname, cid, score = m.groups()
                stage3_scores.append((cname.strip(), int(cid), float(score)))
        if stage3_scores:
            best_c = max(stage3_scores, key=lambda x: x[2])
            best_concept_name, best_concept_id, best_score = best_c


        # #N Entity: status - 엔티티별 큐에서 pop (같은 엔티티가 여러 run에서 반복)
        if entity_key in status_by_entity and status_by_entity[entity_key]:
            test_index, entity_display, mapping_correct = status_by_entity[entity_key].pop(0)
        else:
            # status 없음 (진행 중 블록): entity_key로 CSV 매칭 시 merge_with_csv에서 test_index 보완
            test_index = len(blocks) + 1
            entity_display = entity_name
            mapping_correct = False

        run_idx = _run_at_line(start_idx)
        success = best_concept_id is not None
        blocks.append({
            "test_index": test_index,
            "entity_name": entity_display,
            "entity_key": entity_key,
            "input_domain": domain,
            "ground_truth_concept_id": None,
            "ground_truth_concept_name": None,
            "id": None,
            "success": success,
            "mapping_correct": mapping_correct,
            "best_result_domain": best_domain or domain,
            "best_concept_id": best_concept_id,
            "best_concept_name": best_concept_name,
            "best_score": best_score,
            "stage1_raw": "\n".join(stage1_raw),
            "stage2_raw": "\n".join(stage2_raw),
            "stage3_raw": "\n".join(stage3_raw),
            "run_idx": run_idx,
        })

    return blocks, run_boundaries, num_runs, completed_run_nums


def merge_with_csv(
    blocks: list[dict],
    csv_path: Path,
    entity_col: str = "source_value",
    id_col: str = "index",
    domain_col: str = "domain_id",
    gt_id_col: str = "concept_id",
    gt_name_col: str = "concept_name",
) -> list[dict]:
    """CSV와 병합하여 ground_truth, id 보완."""
    df = pd.read_csv(csv_path)
    # entity_col가 없으면 source_name 시도
    if entity_col not in df.columns and "source_name" in df.columns:
        entity_col = "source_name"

    # CSV 행: index 1-based -> test_index
    csv_by_entity: dict[str, dict] = {}
    for _, row in df.iterrows():
        entity_val = str(row.get(entity_col, "")).strip()
        entity_key = " ".join(entity_val.split())
        idx = row.get(id_col, row.get("index", row.name))
        if pd.isna(idx):
            idx = len(csv_by_entity) + 1
        test_idx = int(idx) if pd.notna(idx) else None
        csv_by_entity[entity_key] = {
            "id": str(int(idx)) if pd.notna(idx) else "N/A",
            "test_index": test_idx,
            "ground_truth_concept_id": int(row[gt_id_col]) if pd.notna(row.get(gt_id_col)) else None,
            "ground_truth_concept_name": str(row.get(gt_name_col, "")).strip() if pd.notna(row.get(gt_name_col)) else None,
            "domain": str(row.get(domain_col, "")).strip() if pd.notna(row.get(domain_col)) else None,
        }

    for b in blocks:
        ek = b.get("entity_key", b["entity_name"])
        ek_norm = " ".join(ek.split())
        if ek_norm in csv_by_entity:
            c = csv_by_entity[ek_norm]
            b["id"] = c["id"]
            b["ground_truth_concept_id"] = c["ground_truth_concept_id"]
            b["ground_truth_concept_name"] = c["ground_truth_concept_name"]
            if c.get("domain") and b.get("input_domain") == "All":
                b["input_domain"] = c["domain"]
            # CSV 매칭 시 test_index 보완 (status 없을 때 또는 검증)
            if c.get("test_index") is not None:
                b["test_index"] = c["test_index"]
        else:
            # test_index로 매칭 시도 (CSV 행 순서 = index)
            for _, row in df.iterrows():
                if int(row.get(id_col, row.get("index", 0))) == b["test_index"]:
                    b["id"] = str(int(row.get(id_col, row.get("index", 0))))
                    b["ground_truth_concept_id"] = int(row[gt_id_col]) if pd.notna(row.get(gt_id_col)) else None
                    b["ground_truth_concept_name"] = str(row.get(gt_name_col, "")).strip() if pd.notna(row.get(gt_name_col)) else None
                    break

        if b.get("id") is None:
            b["id"] = str(b["test_index"])

        # 정답/오답: Ground Truth Concept ID vs Best Concept ID 비교 (로그 status 무시)
        gt_id = b.get("ground_truth_concept_id")
        best_id = b.get("best_concept_id")
        mapping_correct = False
        if gt_id is not None and best_id is not None:
            try:
                mapping_correct = int(gt_id) == int(best_id)
            except (ValueError, TypeError):
                pass
        b["mapping_correct"] = mapping_correct

    return blocks


def save_xlsx_from_log(
    all_results: list[list[dict]],
    output_dir: Path,
    data_type: str,
    timestamp: str,
) -> Path:
    """
    현황 시트(맨 앞) + Run 1~N 상세 시트 저장.
    Stage 열에는 로그 원문(stage1_raw, stage2_raw, stage3_raw) 그대로 저장.
    """
    if not HAS_OPENPYXL:
        raise ImportError("openpyxl required. pip install openpyxl")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"mapping_{data_type}_{timestamp}_from_log.xlsx"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center")

    wb = openpyxl.Workbook()

    # 시트 1: 현황 (맨 앞)
    num_runs = len(all_results)
    summary_headers = SUMMARY_HEADERS[: 8 + min(5, num_runs)]
    ws_summary = wb.active
    ws_summary.title = "현황"
    for col, h in enumerate(summary_headers, 1):
        cell = ws_summary.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align

    by_index: dict[int, list[dict]] = {}
    for run_results in all_results:
        for r in run_results:
            idx = r.get("test_index")
            if idx not in by_index:
                by_index[idx] = []
            by_index[idx].append(r)

    for row_idx, test_index in enumerate(sorted(by_index.keys()), 2):
        rows = by_index[test_index]
        r0 = rows[0]
        ws_summary.cell(row=row_idx, column=1, value=test_index)
        ws_summary.cell(row=row_idx, column=2, value=r0.get("id", "N/A"))
        ws_summary.cell(row=row_idx, column=3, value=r0.get("entity_name", ""))
        ws_summary.cell(row=row_idx, column=4, value=r0.get("input_domain", "All"))
        ws_summary.cell(row=row_idx, column=5, value=r0.get("ground_truth_concept_id", ""))
        ws_summary.cell(row=row_idx, column=6, value=r0.get("ground_truth_concept_name", ""))

        concept_ids = [str(r.get("best_concept_id") or "") for r in rows]
        all_same = len(set(concept_ids)) == 1 if concept_ids else False
        ws_summary.cell(row=row_idx, column=7, value="Y" if all_same else "N")

        correct_count = sum(1 for r in rows if r.get("mapping_correct"))
        correct_display = "Y" if (len(rows) == 1 and correct_count == 1) else ("N" if (len(rows) == 1 and correct_count == 0) else correct_count)
        ws_summary.cell(row=row_idx, column=8, value=correct_display)

        for i, r in enumerate(rows[:5]):
            cid = r.get("best_concept_id") or "N/A"
            cname = r.get("best_concept_name") or "N/A"
            ws_summary.cell(row=row_idx, column=9 + i, value=f"{cname}({cid})")

    for col_letter, w in {"A": 10, "B": 18, "C": 40, "D": 15, "E": 20, "F": 45, "G": 10, "H": 8}.items():
        ws_summary.column_dimensions[col_letter].width = w
    for i in range(5):
        ws_summary.column_dimensions[openpyxl.utils.get_column_letter(9 + i)].width = 50

    # 시트 2~: Run 1 상세, Run 2 상세, ...
    for run_idx, run_results in enumerate(all_results, 1):
        ws = wb.create_sheet(title=f"Run {run_idx} 상세", index=run_idx)

        for col, h in enumerate(XLSX_HEADERS, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align

        for row_idx, r in enumerate(run_results, 2):
            ws.cell(row=row_idx, column=1, value=r.get("test_index", ""))
            ws.cell(row=row_idx, column=2, value=r.get("id", r.get("snuh_id", r.get("note_id", "N/A"))))
            ws.cell(row=row_idx, column=3, value=r.get("entity_name", ""))
            ws.cell(row=row_idx, column=4, value=r.get("input_domain", "All"))
            ws.cell(row=row_idx, column=5, value=r.get("ground_truth_concept_id", ""))
            ws.cell(row=row_idx, column=6, value=r.get("ground_truth_concept_name", ""))
            ws.cell(row=row_idx, column=7, value="성공" if r.get("success") else "실패")

            correct_cell = ws.cell(row=row_idx, column=8, value="정답" if r.get("mapping_correct") else "오답")
            if r.get("mapping_correct"):
                correct_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                correct_cell.font = Font(color="006100")
            else:
                correct_cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                correct_cell.font = Font(color="9C0006")

            ws.cell(row=row_idx, column=9, value=r.get("best_result_domain", "N/A"))
            ws.cell(row=row_idx, column=10, value=r.get("best_concept_id", "N/A"))
            ws.cell(row=row_idx, column=11, value=r.get("best_concept_name", "N/A"))
            ws.cell(row=row_idx, column=12, value=r.get("best_score", 0.0))

            # Stage 1/2/3: 로그 원문 그대로
            stage1_text = r.get("stage1_raw", "후보 없음") or "후보 없음"
            stage2_text = r.get("stage2_raw", "후보 없음") or "후보 없음"
            stage3_text = r.get("stage3_raw", "후보 없음") or "후보 없음"
            ws.cell(row=row_idx, column=13, value=stage1_text)
            ws.cell(row=row_idx, column=14, value=stage2_text)
            ws.cell(row=row_idx, column=15, value=stage3_text)

            for c in range(13, 16):
                ws.cell(row=row_idx, column=c).alignment = Alignment(wrap_text=True, vertical="top")

        widths = {
            "A": 10, "B": 18, "C": 40, "D": 15, "E": 20, "F": 45, "G": 10, "H": 12,
            "I": 18, "J": 15, "K": 45, "L": 12, "M": 70, "N": 70, "O": 85,
        }
        for letter, w in widths.items():
            ws.column_dimensions[letter].width = w
        for rn in range(2, len(run_results) + 2):
            ws.row_dimensions[rn].height = 150

    wb.save(out_path)
    return out_path


def infer_csv_path(log_path: Path, data_type: str = "snuh") -> Path | None:
    """로그 경로/이름에서 data_type 추론 후 DATA_SOURCES의 csv_path 반환."""
    name = log_path.stem  # mapping_snuh_20260303_115235
    if "snuh" in name:
        return Path(DATA_SOURCES["snuh"]["csv_path"])
    if "snomed" in name:
        return Path(DATA_SOURCES["snomed"]["csv_path"])
    return None


def main():
    parser = argparse.ArgumentParser(description="매핑 로그 → Excel 변환 (중간 결과 조회용)")
    parser.add_argument("log_path", type=Path, help="로그 파일 경로")
    parser.add_argument("--csv", type=Path, default=None, help="CSV 경로 (미지정 시 data_type으로 추론)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="출력 Excel 경로 (미지정 시 test_logs/mapping_{type}_{ts}_from_log.xlsx)")
    parser.add_argument("--watch", action="store_true", help="10초마다 로그 재파싱 후 Excel 갱신 (Ctrl+C로 종료)")
    parser.add_argument("--repeat", type=int, default=1, help="num_runs (반복 횟수). 1이면 단일 시트, 2+면 현황+상세 시트")
    args = parser.parse_args()

    log_path = args.log_path
    if not log_path.exists():
        print(f"오류: 로그 파일 없음: {log_path}")
        sys.exit(1)

    csv_path = args.csv or infer_csv_path(log_path)
    if csv_path and not csv_path.exists():
        csv_path = None
        print("경고: CSV 파일 없음. ground_truth, id 없이 진행합니다.")

    data_type = "snuh" if "snuh" in log_path.stem else "snomed"
    # 타임스탬프 추출: mapping_snuh_20260303_115235 -> 20260303_115235
    parts = log_path.stem.split("_")
    timestamp = "_".join(parts[-2:]) if len(parts) >= 2 else "unknown"
    output_dir = log_path.parent

    def run_once():
        print(f"파싱 중: {log_path}")
        blocks, run_boundaries, num_runs, completed_run_nums = parse_log_blocks(log_path)
        print(f"  추출된 블록: {len(blocks)}개, Run: {num_runs}회, 마무리된 Run: {sorted(completed_run_nums) or '없음'}")

        if csv_path:
            blocks = merge_with_csv(blocks, csv_path)
            print(f"  CSV 병합 완료: {csv_path}")

        # 마무리된 run만 필터 (진행 중 run 제외)
        blocks = [b for b in blocks if b.get("run_idx", 1) in completed_run_nums]

        if not blocks:
            print("  마무리된 run 없음. Excel 저장 건너뜀.")
            return None

        # run_idx별로 분할 (데이터 있는 run만)
        run_indices = sorted({b.get("run_idx", 1) for b in blocks})
        if not run_indices or (len(run_indices) == 1 and run_indices[0] == 1 and num_runs == 1):
            all_results = [sorted(blocks, key=lambda x: (x.get("test_index", 0), x.get("entity_name", "")))]
        else:
            all_results = []
            for r in run_indices:
                run_blocks = [b for b in blocks if b.get("run_idx") == r]
                run_blocks.sort(key=lambda x: (x.get("test_index", 0), x.get("entity_name", "")))
                all_results.append(run_blocks)

        out_path = save_xlsx_from_log(all_results, output_dir, data_type, timestamp)
        print(f"저장됨: {out_path} (마무리된 Run {len(run_indices)}개)")
        return out_path

    if args.watch:
        import time
        print("Watch 모드: 10초마다 갱신 (Ctrl+C 종료)")
        while True:
            try:
                run_once()
                time.sleep(10)
            except KeyboardInterrupt:
                print("\n종료")
                break
    else:
        run_once()


if __name__ == "__main__":
    main()
