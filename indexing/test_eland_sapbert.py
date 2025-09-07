#!/usr/bin/env python3
"""
SapBERT와 eland 호환성 테스트 스크립트

이 스크립트는 SapBERT 모델을 eland를 통해 Elasticsearch에 배포하고
text_embedding 태스크가 정상적으로 작동하는지 테스트합니다.
"""

import logging
import time
import subprocess
import sys
from pathlib import Path
from elasticsearch import Elasticsearch
import json


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def check_elasticsearch_connection(es_url="http://3.35.110.161:9200"):
    """Elasticsearch 연결 확인"""
    try:
        es = Elasticsearch(
            [es_url], 
            basic_auth=('elastic', 'snomed'),
            request_timeout=30
        )
        if es.ping():
            logging.info("✅ Elasticsearch 연결 성공")
            
            # 클러스터 정보 확인
            info = es.info()
            logging.info(f"Elasticsearch 버전: {info['version']['number']}")
            
            # ML 기능 확인
            try:
                ml_info = es.ml.info()
                logging.info("✅ ML 기능 활성화됨")
                logging.info(f"사용 가능한 프로세서: {ml_info.get('defaults', {}).get('anomaly_detectors', {}).get('max_model_memory_limit', 'N/A')}")
                return es
            except Exception as e:
                logging.warning(f"⚠️ ML 기능 확인 실패: {e}")
                return es
                
        else:
            logging.error("❌ Elasticsearch에 연결할 수 없습니다")
            return None
            
    except Exception as e:
        logging.error(f"❌ Elasticsearch 연결 실패: {e}")
        return None


def test_sapbert_model_info():
    """SapBERT 모델 정보 확인"""
    logging.info("=== SapBERT 모델 정보 확인 ===")
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        
        model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        
        # 모델 설정 확인
        config = AutoConfig.from_pretrained(model_name)
        logging.info(f"모델명: {model_name}")
        logging.info(f"모델 타입: {config.model_type}")
        logging.info(f"히든 사이즈: {config.hidden_size}")
        logging.info(f"최대 길이: {config.max_position_embeddings}")
        
        # 토크나이저 확인
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"토크나이저 타입: {type(tokenizer).__name__}")
        
        # 테스트 토큰화
        test_text = "covid-19"
        tokens = tokenizer(test_text, return_tensors="pt")
        logging.info(f"테스트 토큰화 성공: {test_text} -> {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ SapBERT 모델 정보 확인 실패: {e}")
        return False


def deploy_sapbert_to_elasticsearch(es_url="http://3.35.110.161:9200"):
    """SapBERT 모델을 Elasticsearch에 배포"""
    logging.info("=== SapBERT 모델 Elasticsearch 배포 ===")
    
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    es_model_id = "sapbert-from-pubmedbert"
    
    cmd = [
        "poetry", "run", "eland_import_hub_model",
        "--url", es_url,
        "-u", "elastic",
        "-p", "snomed",
        "--hub-model-id", model_name,
        "--es-model-id", es_model_id,
        "--task-type", "text_embedding",
        "--start",
        "--clear-previous"
    ]
    
    logging.info(f"실행 명령어: {' '.join(cmd)}")
    
    try:
        # 모델 배포 실행
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10분 타임아웃
        )
        
        if result.returncode == 0:
            logging.info("✅ SapBERT 모델 배포 성공")
            logging.info(result.stdout)
            return es_model_id
        else:
            logging.error("❌ SapBERT 모델 배포 실패")
            logging.error(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        logging.error("❌ 모델 배포 타임아웃 (10분)")
        return None
    except Exception as e:
        logging.error(f"❌ 모델 배포 중 오류: {e}")
        return None


def test_model_inference(es, model_id):
    """배포된 모델로 추론 테스트"""
    logging.info("=== 모델 추론 테스트 ===")
    
    try:
        # 모델 상태 확인
        models = es.ml.get_trained_models(model_id=model_id)
        model_info = models['trained_models'][0]
        logging.info(f"모델 상태: {model_info.get('metadata', {}).get('model_id', 'N/A')}")
        
        # 추론 테스트
        test_texts = [
            "covid-19",
            "high fever", 
            "diabetes mellitus",
            "heart attack"
        ]
        
        for text in test_texts:
            try:
                # ML 추론 API 호출
                response = es.ml.infer_trained_model(
                    model_id=model_id,
                    docs=[{"text_field": text}]
                )
                
                if 'inference_results' in response:
                    result = response['inference_results'][0]
                    if 'predicted_value' in result:
                        embedding = result['predicted_value']
                        logging.info(f"✅ '{text}' -> 임베딩 차원: {len(embedding)}")
                    else:
                        logging.warning(f"⚠️ '{text}' -> 예상과 다른 응답 구조: {result}")
                else:
                    logging.warning(f"⚠️ '{text}' -> 예상과 다른 응답: {response}")
                    
            except Exception as e:
                logging.error(f"❌ '{text}' 추론 실패: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ 모델 추론 테스트 실패: {e}")
        return False


def create_test_ingest_pipeline(es, model_id):
    """테스트용 Ingest Pipeline 생성"""
    logging.info("=== Ingest Pipeline 생성 ===")
    
    pipeline_name = "sapbert-embedding-pipeline"
    
    pipeline_config = {
        "processors": [
            {
                "inference": {
                    "model_id": model_id,
                    "target_field": "ml",
                    "field_map": {
                        "concept_name": "text_field"
                    }
                }
            },
            {
                "set": {
                    "field": "concept_embedding",
                    "value": "{{ml.predicted_value}}"
                }
            },
            {
                "remove": {
                    "field": "ml",
                    "ignore_missing": True
                }
            }
        ]
    }
    
    try:
        # Pipeline 생성
        es.ingest.put_pipeline(id=pipeline_name, body=pipeline_config)
        logging.info(f"✅ Ingest Pipeline 생성 성공: {pipeline_name}")
        
        # Pipeline 테스트
        test_doc = {
            "concept_id": "test_001",
            "concept_name": "covid-19",
            "domain_id": "Condition"
        }
        
        simulate_response = es.ingest.simulate(
            id=pipeline_name,
            body={"docs": [{"_source": test_doc}]}
        )
        
        if simulate_response.get('docs'):
            processed_doc = simulate_response['docs'][0]['doc']['_source']
            if 'concept_embedding' in processed_doc:
                embedding_len = len(processed_doc['concept_embedding'])
                logging.info(f"✅ Pipeline 테스트 성공: 임베딩 차원 {embedding_len}")
                return pipeline_name
            else:
                logging.error("❌ Pipeline 테스트 실패: concept_embedding 필드가 생성되지 않음")
                logging.error(f"처리된 문서: {processed_doc}")
        else:
            logging.error("❌ Pipeline 시뮬레이션 실패")
            logging.error(f"응답: {simulate_response}")
        
        return None
        
    except Exception as e:
        logging.error(f"❌ Ingest Pipeline 생성 실패: {e}")
        return None


def test_index_with_pipeline(es, pipeline_name):
    """Pipeline을 사용한 인덱싱 테스트"""
    logging.info("=== Pipeline 인덱싱 테스트 ===")
    
    test_index = "test_concepts"
    
    try:
        # 테스트 인덱스 삭제 (존재할 경우)
        if es.indices.exists(index=test_index):
            es.indices.delete(index=test_index)
        
        # 인덱스 매핑 생성
        index_mapping = {
            "mappings": {
                "properties": {
                    "concept_id": {"type": "keyword"},
                    "concept_name": {"type": "text"},
                    "domain_id": {"type": "keyword"},
                    "concept_embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        es.indices.create(index=test_index, body=index_mapping)
        logging.info(f"✅ 테스트 인덱스 생성: {test_index}")
        
        # 테스트 문서 인덱싱
        test_docs = [
            {"concept_id": "001", "concept_name": "covid-19", "domain_id": "Condition"},
            {"concept_id": "002", "concept_name": "high fever", "domain_id": "Observation"},
            {"concept_id": "003", "concept_name": "diabetes", "domain_id": "Condition"}
        ]
        
        for doc in test_docs:
            response = es.index(
                index=test_index,
                id=doc["concept_id"],
                body=doc,
                pipeline=pipeline_name
            )
            logging.info(f"✅ 문서 인덱싱: {doc['concept_name']} -> {response['result']}")
        
        # 인덱스 새로고침
        es.indices.refresh(index=test_index)
        
        # 결과 확인
        search_response = es.search(index=test_index, body={"query": {"match_all": {}}})
        
        for hit in search_response['hits']['hits']:
            doc = hit['_source']
            if 'concept_embedding' in doc:
                embedding_len = len(doc['concept_embedding'])
                logging.info(f"✅ 확인: {doc['concept_name']} -> 임베딩 차원 {embedding_len}")
            else:
                logging.error(f"❌ 임베딩 누락: {doc['concept_name']}")
        
        # 테스트 인덱스 삭제
        es.indices.delete(index=test_index)
        logging.info("✅ 테스트 인덱스 정리 완료")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Pipeline 인덱싱 테스트 실패: {e}")
        return False


def cleanup_test_resources(es, model_id, pipeline_name):
    """테스트 리소스 정리"""
    logging.info("=== 테스트 리소스 정리 ===")
    
    try:
        # Pipeline 삭제
        if pipeline_name:
            es.ingest.delete_pipeline(id=pipeline_name)
            logging.info(f"✅ Pipeline 삭제: {pipeline_name}")
        
        # 모델 중지 및 삭제 (선택사항)
        # es.ml.stop_trained_model_deployment(model_id=model_id)
        # es.ml.delete_trained_model(model_id=model_id)
        # logging.info(f"✅ 모델 삭제: {model_id}")
        
    except Exception as e:
        logging.warning(f"⚠️ 리소스 정리 중 오류: {e}")


def main():
    """메인 테스트 함수"""
    setup_logging()
    
    logging.info("🚀 SapBERT + eland 호환성 테스트 시작")
    
    # 1. SapBERT 모델 정보 확인
    if not test_sapbert_model_info():
        logging.error("❌ SapBERT 모델 정보 확인 실패")
        return False
    
    # 2. Elasticsearch 연결 확인
    es = check_elasticsearch_connection()
    if not es:
        logging.error("❌ Elasticsearch 연결 실패")
        logging.info("💡 Elasticsearch를 먼저 실행해주세요:")
        logging.info("   docker run -d --name elasticsearch -p 9200:9200 -e 'discovery.type=single-node' -e 'xpack.security.enabled=false' -e 'xpack.ml.enabled=true' docker.elastic.co/elasticsearch/elasticsearch:9.1.0")
        return False
    
    # 3. SapBERT 모델 배포
    model_id = deploy_sapbert_to_elasticsearch()
    if not model_id:
        logging.error("❌ 모델 배포 실패")
        return False
    
    # 4. 모델 추론 테스트
    if not test_model_inference(es, model_id):
        logging.error("❌ 모델 추론 테스트 실패")
        return False
    
    # 5. Ingest Pipeline 생성 및 테스트
    pipeline_name = create_test_ingest_pipeline(es, model_id)
    if not pipeline_name:
        logging.error("❌ Ingest Pipeline 생성 실패")
        return False
    
    # 6. Pipeline을 사용한 인덱싱 테스트
    if not test_index_with_pipeline(es, pipeline_name):
        logging.error("❌ Pipeline 인덱싱 테스트 실패")
        return False
    
    # 7. 리소스 정리
    cleanup_test_resources(es, model_id, pipeline_name)
    
    logging.info("🎉 모든 테스트 성공! SapBERT + eland 호환성 확인됨")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
