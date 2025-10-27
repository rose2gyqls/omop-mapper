import pandas as pd
import logging
import os
from datetime import datetime
from pathlib import Path
import sys
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

# 프로젝트 루트 추가
sys.path.append('/home/work/skku/hyo/omop-mapper/src')

from omop_mapper.entity_mapping_api import EntityMappingAPI, EntityInput, DomainID
from omop_mapper.elasticsearch_client import ElasticsearchClient

class EntityMappingTester:
    def __init__(self, log_dir: str = "test_logs"):
        """테스터 초기화"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로그 설정
        self.setup_logging()
        
        # Elasticsearch 클라이언트 초기화 (concept-small 인덱스 사용)
        self.es_client = ElasticsearchClient()
        self.es_client.concept_index = "concept-small"
        self.es_client.concept_synonym_index = "concept-small"
        
        # API 초기화
        self.api = EntityMappingAPI(es_client=self.es_client)
        
        # 도메인 매핑
        self.domain_mapping = {
            'Condition': DomainID.CONDITION,
            'Procedure': DomainID.PROCEDURE,
            'Drug': DomainID.DRUG,
            'Observation': DomainID.OBSERVATION,
            'Measurement': DomainID.MEASUREMENT,
            'Period': DomainID.PERIOD,
            'Provider': DomainID.PROVIDER
        }
    
    def setup_logging(self):
        """로깅 설정"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 메인 로그 파일
        log_file = self.log_dir / f"entity_mapping_test_{timestamp}.log"
        
        # 로거 설정
        self.logger = logging.getLogger('entity_mapping_test')
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # API 로거도 같은 핸들러 사용
        api_logger = logging.getLogger('omop_mapper.entity_mapping_api')
        api_logger.setLevel(logging.INFO)
        api_logger.addHandler(file_handler)
        
        self.logger.info(f"로그 파일: {log_file}")
    
    def load_test_data_from_list(self, entity_list: list) -> pd.DataFrame:
        """리스트에서 테스트 데이터 생성"""
        self.logger.info(f"테스트 데이터 생성: {len(entity_list)}개 엔티티")
        
        # 엔티티별 도메인 추정 (기본적으로 Condition으로 설정)
        domain_mapping = {
            'Acute Coronary Syndromes': 'Condition',
            'myocardial ischemia': 'Condition', 
            'chronic coronary disease': 'Condition',
            'non–ST-segment elevation myocardial infarction': 'Condition',
            'breast cancer': 'Condition',
            'type 2 diabetes': 'Condition',
            'hypertension': 'Condition'
        }
        
        test_data = []
        for i, entity_name in enumerate(entity_list):
            domain = domain_mapping.get(entity_name, 'Condition')
            test_data.append({
                'entity_plain_name': entity_name,
                'entity_domain': domain,
                'sheet': 'manual'  # 수동 입력 표시
            })
            self.logger.info(f"  {i+1}. {entity_name} (도메인: {domain})")
        
        df = pd.DataFrame(test_data)
        self.logger.info(f"전체 테스트 데이터: {len(df)}개 엔티티")
        
        return df
    
    def create_entity_input(self, row) -> EntityInput:
        """DataFrame 행에서 EntityInput 생성"""
        entity_name = str(row['entity_plain_name']).strip()
        domain_str = str(row['entity_domain']).strip() if pd.notna(row['entity_domain']) else None
        
        # 도메인 매핑
        domain_id = None
        if domain_str and domain_str in self.domain_mapping:
            domain_id = self.domain_mapping[domain_str]
        
        return EntityInput(
            entity_name=entity_name,
            domain_id=domain_id,
            vocabulary_id=None
        )
    
    def test_single_entity(self, entity_input: EntityInput, test_index: int, sheet: str) -> dict:
        """단일 엔티티 테스트"""
        self.logger.info("=" * 100)
        self.logger.info(f"🧪 테스트 #{test_index} (시트 {sheet}): {entity_input.entity_name}")
        self.logger.info(f"도메인: {entity_input.domain_id}")
        self.logger.info("=" * 100)
        
        try:
            # 매핑 수행
            result = self.api.map_entity(entity_input)
            
            # 단계별 상세 정보 로깅
            stage1_candidates = []
            stage3_candidates = []
            
            if hasattr(self.api, '_last_stage1_candidates') and self.api._last_stage1_candidates:
                stage1_candidates = self.api._last_stage1_candidates
                self.logger.info("📊 1단계 후보군 상세 정보:")
                for i, candidate in enumerate(stage1_candidates[:5], 1):  # 상위 5개만
                    self.logger.info(f"   {i}. {candidate['concept_name']} (ID: {candidate['concept_id']})")
                    self.logger.info(f"      - Elasticsearch 점수: {candidate['elasticsearch_score']:.4f}")
                    self.logger.info(f"      - Standard: {candidate['standard_concept']}")
                    self.logger.info(f"      - Vocabulary: {candidate['vocabulary_id']}")
            
            if hasattr(self.api, '_last_rerank_candidates') and self.api._last_rerank_candidates:
                stage3_candidates = self.api._last_rerank_candidates
                self.logger.info("📊 3단계 후보군 상세 정보:")
                for i, candidate in enumerate(stage3_candidates[:5], 1):  # 상위 5개만
                    self.logger.info(f"   {i}. {candidate['concept_name']} (ID: {candidate['concept_id']})")
                    self.logger.info(f"      - 텍스트 유사도: {candidate['text_similarity']:.4f}")
                    self.logger.info(f"      - 의미적 유사도: {candidate['semantic_similarity']:.4f}")
                    self.logger.info(f"      - 최종 점수: {candidate['final_score']:.4f}")
                    self.logger.info(f"      - Vocabulary: {candidate['vocabulary_id']}")
            
            # 결과 정리
            test_result = {
                'test_index': test_index,
                'sheet': sheet,
                'entity_name': entity_input.entity_name,
                'domain_id': str(entity_input.domain_id) if entity_input.domain_id else None,
                'success': result is not None,
                'mapped_concept_id': result.mapped_concept_id if result else None,
                'mapped_concept_name': result.mapped_concept_name if result else None,
                'mapping_score': result.mapping_score if result else 0.0,
                'mapping_confidence': result.mapping_confidence if result else None,
                'mapping_method': result.mapping_method if result else None,
                'alternative_concepts_count': len(result.alternative_concepts) if result and result.alternative_concepts else 0,
                'stage1_candidates': stage1_candidates,
                'stage3_candidates': stage3_candidates
            }
            
            if result:
                self.logger.info(f"✅ 매핑 성공!")
                self.logger.info(f"   - 매핑된 컨셉: {result.mapped_concept_name} (ID: {result.mapped_concept_id})")
                self.logger.info(f"   - 매핑 점수: {result.mapping_score:.4f}")
                self.logger.info(f"   - 매핑 신뢰도: {result.mapping_confidence}")
                self.logger.info(f"   - 매핑 방법: {result.mapping_method}")
                self.logger.info(f"   - Vocabulary: {result.vocabulary_id}")
                if result.alternative_concepts:
                    self.logger.info(f"   - 대안 개수: {len(result.alternative_concepts)}개")
                    for i, alt in enumerate(result.alternative_concepts[:3], 1):  # 상위 3개 대안
                        self.logger.info(f"     {i}. {alt['concept_name']} (ID: {alt['concept_id']}, 점수: {alt['score']:.4f})")
            else:
                self.logger.info(f"❌ 매핑 실패")
                
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ 테스트 오류: {str(e)}")
            return {
                'test_index': test_index,
                'sheet': sheet,
                'entity_name': entity_input.entity_name,
                'domain_id': str(entity_input.domain_id) if entity_input.domain_id else None,
                'success': False,
                'error': str(e),
                'mapped_concept_id': None,
                'mapped_concept_name': None,
                'mapping_score': 0.0,
                'mapping_confidence': None,
                'mapping_method': None,
                'alternative_concepts_count': 0,
                'stage1_candidates': [],
                'stage3_candidates': []
            }
    
    def run_test_with_entities(self, entity_list: list, max_entities: int = None):
        """엔티티 리스트로 테스트 실행"""
        self.logger.info("🚀 Entity Mapping API 테스트 시작")
        self.logger.info(f"테스트 엔티티 리스트: {len(entity_list)}개")
        
        # 데이터 생성
        test_data = self.load_test_data_from_list(entity_list)
        
        if max_entities:
            test_data = test_data.head(max_entities)
            self.logger.info(f"테스트 제한: 최대 {max_entities}개 엔티티")
        
        # 테스트 결과 저장
        test_results = []
        successful_tests = 0
        
        for idx, row in test_data.iterrows():
            try:
                entity_input = self.create_entity_input(row)
                result = self.test_single_entity(entity_input, idx + 1, row['sheet'])
                test_results.append(result)
                
                if result['success']:
                    successful_tests += 1
                    
            except Exception as e:
                self.logger.error(f"테스트 #{idx + 1} 처리 오류: {str(e)}")
                continue
        
        # 결과 요약
        total_tests = len(test_results)
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.logger.info("=" * 100)
        self.logger.info("📊 테스트 결과 요약")
        self.logger.info("=" * 100)
        self.logger.info(f"총 테스트: {total_tests}개")
        self.logger.info(f"성공: {successful_tests}개")
        self.logger.info(f"실패: {total_tests - successful_tests}개")
        self.logger.info(f"성공률: {success_rate:.2f}%")
        
        # 엔티티별 요약
        for i, result in enumerate(test_results, 1):
            status = "✅ 성공" if result['success'] else "❌ 실패"
            self.logger.info(f"  {i}. {result['entity_name']}: {status}")
            if result['success']:
                self.logger.info(f"     -> {result['mapped_concept_name']} (점수: {result['mapping_score']:.4f})")
        
        # 결과를 CSV와 XLSX로 저장
        self.save_results_to_csv(test_results)
        self.save_results_to_xlsx(test_results)
        
        return test_results
    
    def save_results_to_csv(self, test_results: list):
        """테스트 결과를 CSV 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.log_dir / f"test_results_{timestamp}.csv"
        
        # CSV용 데이터 정리 (복잡한 객체 제거)
        csv_results = []
        for result in test_results:
            csv_result = {k: v for k, v in result.items() 
                         if k not in ['stage1_candidates', 'stage3_candidates']}
            csv_results.append(csv_result)
        
        df_results = pd.DataFrame(csv_results)
        df_results.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"📄 테스트 결과 CSV 저장: {csv_file}")
    
    def save_results_to_xlsx(self, test_results: list):
        """테스트 결과를 XLSX 파일로 저장 (stage1, stage3 후보군을 열로 분리)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_file = self.log_dir / f"test_results_detailed_{timestamp}.xlsx"
        
        # 엑셀 워크북 생성
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Detailed Results"
        
        # 통합 상세 시트 생성
        self._create_integrated_detail_sheet(ws, test_results)
        
        # 파일 저장
        wb.save(xlsx_file)
        self.logger.info(f"📊 테스트 결과 XLSX 저장: {xlsx_file}")
    
    def _create_integrated_detail_sheet(self, ws, test_results):
        """통합 상세 시트 생성 (모든 엔티티를 하나의 시트에)"""
        
        # 헤더 설정
        headers = [
            "Test Index", "Entity Name", "Domain", "Success", 
            "Mapped Concept ID", "Mapped Concept Name", 
            "Mapping Score", "Mapping Confidence", "Mapping Method",
            "Stage1 Candidates", "Stage3 Candidates"
        ]
        
        # 헤더 스타일
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")
        
        # 헤더 작성
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
        
        # 데이터 작성
        for row, result in enumerate(test_results, 2):
            ws.cell(row=row, column=1, value=result['test_index'])
            ws.cell(row=row, column=2, value=result['entity_name'])
            ws.cell(row=row, column=3, value=result['domain_id'])
            ws.cell(row=row, column=4, value="성공" if result['success'] else "실패")
            ws.cell(row=row, column=5, value=result['mapped_concept_id'])
            ws.cell(row=row, column=6, value=result['mapped_concept_name'])
            ws.cell(row=row, column=7, value=result['mapping_score'])
            ws.cell(row=row, column=8, value=result['mapping_confidence'])
            ws.cell(row=row, column=9, value=result['mapping_method'])
            
            # Stage1 후보군 정보를 문자열로 변환
            stage1_text = self._format_candidates_for_cell(result.get('stage1_candidates', []), 'stage1')
            ws.cell(row=row, column=10, value=stage1_text)
            
            # Stage3 후보군 정보를 문자열로 변환
            stage3_text = self._format_candidates_for_cell(result.get('stage3_candidates', []), 'stage3')
            ws.cell(row=row, column=11, value=stage3_text)
            
            # 셀 스타일 설정 (텍스트 줄바꿈 허용)
            for col in range(10, 12):  # Stage1, Stage3 열
                cell = ws.cell(row=row, column=col)
                cell.alignment = Alignment(wrap_text=True, vertical='top')
        
        # 열 너비 설정
        column_widths = {
            'A': 10,  # Test Index
            'B': 30,  # Entity Name
            'C': 15,  # Domain
            'D': 10,  # Success
            'E': 15,  # Mapped Concept ID
            'F': 40,  # Mapped Concept Name
            'G': 15,  # Mapping Score
            'H': 15,  # Mapping Confidence
            'I': 20,  # Mapping Method
            'J': 60,  # Stage1 Candidates
            'K': 80   # Stage3 Candidates
        }
        
        for col_letter, width in column_widths.items():
            ws.column_dimensions[col_letter].width = width
        
        # 행 높이 자동 조정 (후보군 정보가 많은 경우)
        for row_num in range(2, len(test_results) + 2):
            ws.row_dimensions[row_num].height = 120  # 충분한 높이 설정
    
    def _format_candidates_for_cell(self, candidates, stage_type):
        """후보군 정보를 엑셀 셀용 텍스트로 포맷팅"""
        if not candidates:
            return "후보 없음"
        
        lines = []
        for i, candidate in enumerate(candidates[:5], 1):  # 상위 5개만 표시
            if stage_type == 'stage1':
                line = f"{i}. {candidate.get('concept_name', 'N/A')} (ID: {candidate.get('concept_id', 'N/A')})\n"
                line += f"   ES점수: {candidate.get('elasticsearch_score', 0):.4f}, "
                line += f"Standard: {candidate.get('standard_concept', 'N/A')}, "
                line += f"Vocab: {candidate.get('vocabulary_id', 'N/A')}"
            else:  # stage3
                line = f"{i}. {candidate.get('concept_name', 'N/A')} (ID: {candidate.get('concept_id', 'N/A')})\n"
                line += f"   텍스트: {candidate.get('text_similarity', 0):.4f}, "
                line += f"의미적: {candidate.get('semantic_similarity', 0):.4f}, "
                line += f"최종: {candidate.get('final_score', 0):.4f}\n"
                line += f"   Standard: {candidate.get('standard_concept', 'N/A')}, "
                line += f"Vocab: {candidate.get('vocabulary_id', 'N/A')}"
            
            lines.append(line)
        
        return "\n\n".join(lines)

def main():
    """메인 함수"""
    tester = EntityMappingTester()
    
    # 테스트할 엔티티 리스트
    test_entities = test_entities = [
        'MI with nonobstructive coronary artery disease',
        'ST-segment elevation myocardial infarction',
        'US Food and Drug Administration',
        'acute coronary syndromes',
        'acute myocardial infarction',
        'adrenal incidentaloma',
        'adrenal vein sampling',
        'aldosterone-producing adenoma',
        'aldosterone-to-renin ratio',
        'angiotensin receptor blocker',
        'angiotensin-converting enzyme inhibitor',
        'atherosclerotic cardiovascular disease',
        'atrial fibrillation',
        'blood pressure',
        'cardiac intensive care unit',
        'cardiac rehabilitation',
        'cardiac troponin',
        'cardiovascular',
        'cardiovascular disease',
        'chronic coronary disease',
        'clinical decision pathway',
        'computed tomography',
        'confidence interval',
        'coronary artery bypass grafting',
        'coronary artery disease',
        'diastolic blood pressure',
        'direct oral anticoagulant',
        'dual antiplatelet therapy',
        'electrocardiogram',
        'first medical contact',
        'fractional flow reserve',
        'glucagon-like peptide-1',
        'glucocorticoid-remediable aldosteronism',
        'hazard ratio',
        'heart failure',
        'high-sensitivity cardiac troponin',
        'hypertension',
        'idiopathic hyperaldosteronism',
        'implantable cardioverter-defibrillator',
        'intra-aortic balloon pump',
        'intravascular ultrasound',
        'left ventricular',
        'left ventricular ejection fraction',
        'left ventricular hypertrophy',
        'low-density lipoprotein',
        'low-density lipoprotein cholesterol',
        'major adverse cardiovascular event',
        'mechanical circulatory support',
        'mineralocorticoid receptor',
        'mineralocorticoid receptor antagonist',
        'multivessel disease',
        'non–ST-segment elevation ACS',
        'non–ST-segment elevation myocardial infarction',
        'odds ratio',
        'optical coherence tomography',
        'percutaneous coronary intervention',
        'plasma aldosterone concentration',
        'plasma renin activity',
        'primary aldosteronism',
        'proprotein convertase subtilisin/kexin type 9',
        'primary percutaneous coronary intervention',
        'proton pump inhibitor',
        'quality of life',
        'randomized controlled trial',
        'relative risk',
        'renin-angiotensin system',
        'return of spontaneous circulation',
        'sodium-glucose cotransporter-2',
        'subclinical hypercortisolism',
        'systolic blood pressure',
        'unfractionated heparin',
        'venoarterial extracorporeal membrane oxygenation',
        'myocardial ischemia',
        'breast cancer',
        'type 2 diabetes',
        # --- 추가 (여기서부터 확장) ---
        'chronic kidney disease',
        'glomerular filtration rate',
        'end-stage renal disease',
        'atrial flutter',
        'ventricular tachycardia',
        'ventricular fibrillation',
        'sudden cardiac death',
        'ischemic stroke',
        'hemorrhagic stroke',
        'pulmonary embolism',
        'deep vein thrombosis',
        'chronic obstructive pulmonary disease',
        'obstructive sleep apnea',
        'acute respiratory distress syndrome',
        'body mass index',
        'fasting plasma glucose',
        'oral glucose tolerance test',
        'glycated hemoglobin',
        'insulin resistance',
        'metabolic syndrome',
        'nonalcoholic fatty liver disease',
        'nonalcoholic steatohepatitis',
        'hepatocellular carcinoma',
        'prostate cancer',
        'colorectal cancer'
    ]
    
    results = tester.run_test_with_entities(test_entities)
    
    print(f"\n✅ 테스트 완료! 로그는 {tester.log_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
