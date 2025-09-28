import pandas as pd
import logging
import os
from datetime import datetime
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.append('/home/work/skku/hyo/omop-mapper/src')

from omop_mapper.entity_mapping_api import EntityMappingAPI, EntityInput, DomainID

class EntityMappingTester:
    def __init__(self, log_dir: str = "test_logs"):
        """테스터 초기화"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로그 설정
        self.setup_logging()
        
        # API 초기화
        self.api = EntityMappingAPI()
        
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
    
    def load_test_data(self, excel_path: str) -> pd.DataFrame:
        """테스트 데이터 로드"""
        self.logger.info(f"테스트 데이터 로드: {excel_path}")
        
        all_data = []
        
        for sheet in ['10', '11', '12']:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            
            # NaN이 아닌 entity_plain_name만 필터링
            valid_entities = df.dropna(subset=['entity_plain_name'])
            
            self.logger.info(f"시트 {sheet}: 총 {len(df)}행, 유효한 엔티티 {len(valid_entities)}개")
            
            # 시트 정보 추가
            valid_entities = valid_entities.copy()
            valid_entities['sheet'] = sheet
            
            all_data.append(valid_entities)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"전체 테스트 데이터: {len(combined_data)}개 엔티티")
        
        return combined_data
    
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
            if hasattr(self.api, '_last_rerank_candidates') and self.api._last_rerank_candidates:
                self.logger.info("📊 3단계 후보군 상세 정보:")
                for i, candidate in enumerate(self.api._last_rerank_candidates[:5], 1):  # 상위 5개만
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
                'alternative_concepts_count': len(result.alternative_concepts) if result and result.alternative_concepts else 0
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
                'alternative_concepts_count': 0
            }
    
    def run_test(self, excel_path: str, max_entities: int = None):
        """전체 테스트 실행"""
        self.logger.info("🚀 Entity Mapping API 테스트 시작")
        self.logger.info(f"테스트 파일: {excel_path}")
        
        # 데이터 로드
        test_data = self.load_test_data(excel_path)
        
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
        
        # 시트별 요약
        for sheet in ['10', '11', '12']:
            sheet_results = [r for r in test_results if r['sheet'] == sheet]
            sheet_success = len([r for r in sheet_results if r['success']])
            sheet_total = len(sheet_results)
            sheet_rate = (sheet_success / sheet_total * 100) if sheet_total > 0 else 0
            self.logger.info(f"시트 {sheet}: {sheet_success}/{sheet_total} ({sheet_rate:.2f}%)")
        
        # 결과를 CSV로 저장
        self.save_results_to_csv(test_results)
        
        return test_results
    
    def save_results_to_csv(self, test_results: list):
        """테스트 결과를 CSV 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.log_dir / f"test_results_{timestamp}.csv"
        
        df_results = pd.DataFrame(test_results)
        df_results.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"📄 테스트 결과 CSV 저장: {csv_file}")

def main():
    """메인 함수"""
    tester = EntityMappingTester()
    excel_path = "/home/work/skku/hyo/omop-mapper/data/entity_sample.xlsx"
    results = tester.run_test(excel_path)
    
    print(f"\n✅ 테스트 완료! 로그는 {tester.log_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
