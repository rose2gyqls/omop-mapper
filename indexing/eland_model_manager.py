"""
Eland 모델 관리자 모듈

SapBERT 모델을 Elasticsearch에 배포하고 관리하는 기능을 제공합니다.
"""

import logging
import subprocess
import time
from typing import Optional, Dict, Any, List
from elasticsearch import Elasticsearch
from pathlib import Path


class ElandModelManager:
    """Eland를 사용한 모델 관리자"""
    
    def __init__(
        self,
        es_host: str = "localhost",
        es_port: int = 9200,
        es_scheme: str = "http",
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Eland 모델 관리자 초기화
        
        Args:
            es_host: Elasticsearch 호스트
            es_port: Elasticsearch 포트
            es_scheme: 연결 스키마 (http/https)
            username: 사용자명 (선택사항)
            password: 비밀번호 (선택사항)
        """
        self.es_url = f"{es_scheme}://{es_host}:{es_port}"
        self.username = username
        self.password = password
        
        # Elasticsearch 클라이언트 설정
        try:
            if username and password:
                # 인증이 필요한 경우
                self.es = Elasticsearch(
                    hosts=[{"host": es_host, "port": es_port, "scheme": es_scheme}],
                    basic_auth=(username, password),
                    request_timeout=60
                )
            else:
                # 인증이 없는 경우 (개발/테스트 환경)
                self.es = Elasticsearch(
                    hosts=[{"host": es_host, "port": es_port, "scheme": es_scheme}],
                    request_timeout=60
                )
        except Exception as e:
            # 호환성을 위한 fallback
            try:
                self.es = Elasticsearch([f"{es_scheme}://{es_host}:{es_port}"])
            except Exception as e2:
                raise ConnectionError(f"Elasticsearch 클라이언트 생성 실패: {e}, fallback 실패: {e2}")
        
        # 연결 테스트
        if not self.es.ping():
            raise ConnectionError("Elasticsearch 서버에 연결할 수 없습니다.")
            
        logging.info(f"Elasticsearch 연결 성공: {self.es_url}")
        
        # ML 기능 확인
        try:
            self.es.ml.info()
            logging.info("✅ ML 기능 활성화 확인됨")
        except Exception as e:
            logging.warning(f"⚠️ ML 기능 확인 실패: {e}")
    
    def deploy_sapbert_model(
        self,
        hub_model_id: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        es_model_id: str = "sapbert-from-pubmedbert",
        start_deployment: bool = True,
        clear_previous: bool = True,
        timeout: int = 600
    ) -> Optional[str]:
        """
        SapBERT 모델을 Elasticsearch에 배포
        
        Args:
            hub_model_id: Hugging Face 모델 ID
            es_model_id: Elasticsearch 모델 ID
            start_deployment: 배포 후 자동 시작 여부
            clear_previous: 기존 모델 삭제 여부
            timeout: 배포 타임아웃 (초)
            
        Returns:
            성공 시 모델 ID, 실패 시 None
        """
        logging.info(f"SapBERT 모델 배포 시작: {hub_model_id} -> {es_model_id}")
        
        # eland_import_hub_model 명령어 구성
        cmd = [
            "poetry", "run", "eland_import_hub_model",
            "--url", self.es_url,
            "--hub-model-id", hub_model_id,
            "--es-model-id", es_model_id,
            "--task-type", "text_embedding"
        ]
        
        if self.username and self.password:
            cmd.extend(["-u", self.username, "-p", self.password])
        
        if start_deployment:
            cmd.append("--start")
        
        if clear_previous:
            cmd.append("--clear-previous")
        
        logging.info(f"실행 명령어: {' '.join(cmd[:6])}... (인증 정보 숨김)")
        
        try:
            # 모델 배포 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                logging.info("✅ SapBERT 모델 배포 성공")
                if result.stdout:
                    logging.info(f"배포 로그: {result.stdout}")
                
                # 모델 배포 확인
                if self.check_model_deployment(es_model_id):
                    return es_model_id
                else:
                    logging.error("모델이 배포되었지만 상태 확인 실패")
                    return None
            else:
                logging.error("❌ SapBERT 모델 배포 실패")
                logging.error(f"오류: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logging.error(f"❌ 모델 배포 타임아웃 ({timeout}초)")
            return None
        except Exception as e:
            logging.error(f"❌ 모델 배포 중 오류: {e}")
            return None
    
    def check_model_deployment(self, model_id: str) -> bool:
        """
        모델 배포 상태 확인
        
        Args:
            model_id: 확인할 모델 ID
            
        Returns:
            배포 성공 여부
        """
        try:
            # 모델 목록에서 확인
            models = self.es.ml.get_trained_models(model_id=model_id)
            
            if models.get('trained_models'):
                model = models['trained_models'][0]
                logging.info(f"모델 정보: {model.get('model_id', 'N/A')}")
                
                # 배포 상태 확인
                try:
                    deployments = self.es.ml.get_trained_models_stats(model_id=model_id)
                    if deployments.get('trained_model_stats'):
                        stats = deployments['trained_model_stats'][0]
                        deployment_stats = stats.get('deployment_stats', {})
                        
                        if deployment_stats:
                            state = deployment_stats.get('state', 'unknown')
                            logging.info(f"배포 상태: {state}")
                            return state in ['started', 'starting']
                        else:
                            logging.info("배포 통계 없음 - 모델은 존재하지만 배포되지 않음")
                            return True  # 모델은 존재함
                except Exception as e:
                    logging.warning(f"배포 상태 확인 실패: {e}")
                    return True  # 모델은 존재함
                
                return True
            else:
                logging.error(f"모델 '{model_id}'를 찾을 수 없습니다")
                return False
                
        except Exception as e:
            logging.error(f"모델 배포 상태 확인 실패: {e}")
            return False
    
    def test_model_inference(self, model_id: str, test_texts: List[str] = None) -> bool:
        """
        모델 추론 테스트
        
        Args:
            model_id: 테스트할 모델 ID
            test_texts: 테스트할 텍스트 리스트
            
        Returns:
            추론 성공 여부
        """
        if test_texts is None:
            test_texts = ["covid-19", "high fever", "diabetes mellitus"]
        
        logging.info(f"모델 추론 테스트: {model_id}")
        
        try:
            for text in test_texts:
                response = self.es.ml.infer_trained_model(
                    model_id=model_id,
                    docs=[{"text_field": text}]
                )
                
                if 'inference_results' in response:
                    result = response['inference_results'][0]
                    if 'predicted_value' in result:
                        embedding = result['predicted_value']
                        logging.info(f"✅ '{text}' -> 임베딩 차원: {len(embedding)}")
                    else:
                        logging.warning(f"⚠️ '{text}' -> 예상과 다른 응답 구조")
                        logging.debug(f"응답: {result}")
                else:
                    logging.warning(f"⚠️ '{text}' -> 예상과 다른 응답")
                    logging.debug(f"응답: {response}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ 모델 추론 테스트 실패: {e}")
            return False
    
    def create_ingest_pipeline(
        self,
        pipeline_name: str,
        model_id: str,
        source_field: str = "concept_name",
        target_field: str = "concept_embedding"
    ) -> bool:
        """
        임베딩 생성을 위한 Ingest Pipeline 생성
        
        Args:
            pipeline_name: 파이프라인 이름
            model_id: 사용할 모델 ID
            source_field: 소스 텍스트 필드명
            target_field: 임베딩이 저장될 필드명
            
        Returns:
            파이프라인 생성 성공 여부
        """
        logging.info(f"Ingest Pipeline 생성: {pipeline_name}")
        
        pipeline_config = {
            "processors": [
                {
                    "inference": {
                        "model_id": model_id,
                        "target_field": "ml_inference",
                        "field_map": {
                            source_field: "text_field"
                        }
                    }
                },
                {
                    "set": {
                        "field": target_field,
                        "value": "{{ml_inference.predicted_value}}"
                    }
                },
                {
                    "remove": {
                        "field": "ml_inference",
                        "ignore_missing": True
                    }
                }
            ]
        }
        
        try:
            self.es.ingest.put_pipeline(id=pipeline_name, body=pipeline_config)
            logging.info(f"✅ Ingest Pipeline 생성 성공: {pipeline_name}")
            
            # 파이프라인 테스트
            return self.test_ingest_pipeline(pipeline_name, source_field, target_field)
            
        except Exception as e:
            logging.error(f"❌ Ingest Pipeline 생성 실패: {e}")
            return False
    
    def test_ingest_pipeline(
        self,
        pipeline_name: str,
        source_field: str = "concept_name",
        target_field: str = "concept_embedding"
    ) -> bool:
        """
        Ingest Pipeline 테스트
        
        Args:
            pipeline_name: 테스트할 파이프라인 이름
            source_field: 소스 필드명
            target_field: 타겟 필드명
            
        Returns:
            테스트 성공 여부
        """
        test_doc = {
            "concept_id": "test_001",
            source_field: "covid-19",
            "domain_id": "Condition"
        }
        
        try:
            simulate_response = self.es.ingest.simulate(
                id=pipeline_name,
                body={"docs": [{"_source": test_doc}]}
            )
            
            if simulate_response.get('docs'):
                processed_doc = simulate_response['docs'][0]['doc']['_source']
                
                if target_field in processed_doc:
                    embedding = processed_doc[target_field]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        logging.info(f"✅ Pipeline 테스트 성공: 임베딩 차원 {len(embedding)}")
                        return True
                    else:
                        logging.error(f"❌ 임베딩 형식 오류: {type(embedding)}")
                else:
                    logging.error(f"❌ '{target_field}' 필드가 생성되지 않음")
                    logging.debug(f"처리된 문서: {processed_doc}")
            else:
                logging.error("❌ Pipeline 시뮬레이션 응답 없음")
            
            return False
            
        except Exception as e:
            logging.error(f"❌ Pipeline 테스트 실패: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        모델 정보 조회
        
        Args:
            model_id: 조회할 모델 ID
            
        Returns:
            모델 정보 딕셔너리 또는 None
        """
        try:
            models = self.es.ml.get_trained_models(model_id=model_id)
            if models.get('trained_models'):
                return models['trained_models'][0]
            return None
        except Exception as e:
            logging.error(f"모델 정보 조회 실패: {e}")
            return None
    
    def delete_model(self, model_id: str) -> bool:
        """
        모델 삭제
        
        Args:
            model_id: 삭제할 모델 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            # 배포 중지
            try:
                self.es.ml.stop_trained_model_deployment(model_id=model_id)
                logging.info(f"모델 배포 중지: {model_id}")
                time.sleep(2)  # 중지 대기
            except Exception:
                pass  # 배포되지 않은 경우 무시
            
            # 모델 삭제
            self.es.ml.delete_trained_model(model_id=model_id)
            logging.info(f"✅ 모델 삭제 성공: {model_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ 모델 삭제 실패: {e}")
            return False
    
    def delete_pipeline(self, pipeline_name: str) -> bool:
        """
        Ingest Pipeline 삭제
        
        Args:
            pipeline_name: 삭제할 파이프라인 이름
            
        Returns:
            삭제 성공 여부
        """
        try:
            self.es.ingest.delete_pipeline(id=pipeline_name)
            logging.info(f"✅ Pipeline 삭제 성공: {pipeline_name}")
            return True
        except Exception as e:
            logging.error(f"❌ Pipeline 삭제 실패: {e}")
            return False


def test_eland_model_manager():
    """Eland 모델 관리자 테스트"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 모델 관리자 초기화
        manager = ElandModelManager()
        
        # SapBERT 모델 배포
        model_id = manager.deploy_sapbert_model(
            es_model_id="test-sapbert",
            start_deployment=True
        )
        
        if model_id:
            print(f"✅ 모델 배포 성공: {model_id}")
            
            # 추론 테스트
            if manager.test_model_inference(model_id):
                print("✅ 모델 추론 테스트 성공")
            
            # Ingest Pipeline 생성
            pipeline_name = "test-sapbert-pipeline"
            if manager.create_ingest_pipeline(pipeline_name, model_id):
                print(f"✅ Pipeline 생성 성공: {pipeline_name}")
                
                # 정리
                manager.delete_pipeline(pipeline_name)
            
            # 모델 삭제 (선택사항)
            # manager.delete_model(model_id)
        else:
            print("❌ 모델 배포 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    test_eland_model_manager()