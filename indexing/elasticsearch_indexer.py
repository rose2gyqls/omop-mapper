"""
Elasticsearch 인덱서 모듈

OMOP CDM CONCEPT 데이터를 Elasticsearch에 인덱싱하는 기능을 제공합니다.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json


class ConceptElasticsearchIndexer:
    """CONCEPT 데이터를 위한 Elasticsearch 인덱서"""
    
    def __init__(
        self,
        es_host: str = "localhost",
        es_port: int = 9200,
        es_scheme: str = "http",
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_name: str = "concepts",
        include_embeddings: bool = True
    ):
        """
        Elasticsearch 인덱서 초기화
        
        Args:
            es_host: Elasticsearch 호스트
            es_port: Elasticsearch 포트
            es_scheme: 연결 스키마 (http/https)
            username: 사용자명 (선택사항)
            password: 비밀번호 (선택사항)
            index_name: 인덱스명
        """
        self.index_name = index_name
        self.include_embeddings = include_embeddings
        
        # Elasticsearch 클라이언트 설정
        try:
            if username and password:
                # 인증이 필요한 경우
                self.es = Elasticsearch(
                    hosts=[{"host": es_host, "port": es_port, "scheme": es_scheme}],
                    basic_auth=(username, password),
                    request_timeout=120,
                    retry_on_timeout=True,
                    max_retries=3
                )
            else:
                # 인증이 없는 경우 (개발/테스트 환경)
                self.es = Elasticsearch(
                    hosts=[{"host": es_host, "port": es_port, "scheme": es_scheme}],
                    request_timeout=120,
                    retry_on_timeout=True,
                    max_retries=3
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
            
        logging.info(f"Elasticsearch 연결 성공: {es_host}:{es_port}")
    

    def create_index(self, delete_if_exists: bool = False) -> bool:
        """
        CONCEPT 인덱스 생성
        
        Args:
            delete_if_exists: 기존 인덱스가 있을 경우 삭제 여부
            
        Returns:
            인덱스 생성 성공 여부
        """
        try:
            # 기존 인덱스 확인 및 삭제
            if self.es.indices.exists(index=self.index_name):
                if delete_if_exists:
                    logging.info(f"기존 인덱스 삭제 중: {self.index_name}")
                    self.es.indices.delete(index=self.index_name)
                else:
                    logging.info(f"인덱스가 이미 존재합니다: {self.index_name}")
                    return True
            
            # 인덱스 매핑 설정
            index_mapping = {
                "mappings": {
                    "properties": {
                        "concept_id": {"type": "keyword"},
                        "concept_name": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword"
                                }
                            }
                        },
                        "domain_id": {"type": "keyword"},
                        "vocabulary_id": {"type": "keyword"},
                        "concept_class_id": {"type": "keyword"},
                        "standard_concept": {"type": "keyword"},
                        "concept_code": {"type": "keyword"},
                        "valid_start_date": {"type": "date", "format": "yyyyMMdd"},
                        "valid_end_date": {"type": "date", "format": "yyyyMMdd"},
                        "invalid_reason": {"type": "keyword"}
                    }
                },
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 5,
                    "refresh_interval": "30s",
                    "index.write.wait_for_active_shards": "1",
                    "index.max_result_window": 50000
                }
            }

            # 임베딩을 포함하는 경우에만 매핑 추가
            if self.include_embeddings:
                index_mapping["mappings"]["properties"]["concept_embedding"] = {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                }
            
            # 인덱스 생성
            self.es.indices.create(index=self.index_name, body=index_mapping)
            logging.info(f"인덱스 생성 완료: {self.index_name}")
            
            # 인덱스 상태 확인
            import time
            time.sleep(2)  # 간단한 대기
            try:
                health = self.es.cluster.health(index=self.index_name, wait_for_status="yellow", timeout="10s")
                logging.info(f"인덱스 상태: {health['status']}")
            except Exception as e:
                logging.warning(f"인덱스 상태 확인 중 오류 (계속 진행): {e}")
            
            return True
            
        except Exception as e:
            logging.error(f"인덱스 생성 실패: {e}")
            return False
    
    def index_concepts(
        self,
        concepts_data: List[Dict[str, Any]],
        batch_size: int = 1000,
        show_progress: bool = True,
        pipeline: Optional[str] = None,
        lowercase_concept_name: bool = False
    ) -> bool:
        """
        CONCEPT 데이터를 Elasticsearch에 인덱싱
        
        Args:
            concepts_data: 인덱싱할 CONCEPT 데이터 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부
            pipeline: Ingest Pipeline 이름 (선택사항)
            lowercase_concept_name: concept_name을 소문자로 변환 여부
            
        Returns:
            인덱싱 성공 여부
        """
        if not concepts_data:
            logging.warning("인덱싱할 데이터가 없습니다.")
            return True
        
        try:
            # 배치 단위로 인덱싱
            total_indexed = 0
            failed_docs = []
            error_count = 0
            
            for i in range(0, len(concepts_data), batch_size):
                batch_data = concepts_data[i:i + batch_size]
                
                # Elasticsearch bulk 형식으로 변환
                actions = []
                for doc in batch_data:
                    # concept_name을 소문자로 변환 (옵션)
                    if lowercase_concept_name and "concept_name" in doc and doc["concept_name"]:
                        doc = doc.copy()  # 원본 데이터 보존
                        doc["concept_name"] = doc["concept_name"].lower()
                    # 임베딩 비활성화 시 필드 제거
                    if not self.include_embeddings and "concept_embedding" in doc:
                        doc = doc.copy()
                        doc.pop("concept_embedding", None)
                    
                    action = {
                        "_index": self.index_name,
                        "_id": doc["concept_id"],  # concept_id를 문서 ID로 사용
                        "_source": doc
                    }
                    actions.append(action)
                
                # 배치 인덱싱 실행
                bulk_params = {
                    "client": self.es,
                    "actions": actions,
                    "index": self.index_name,
                    "chunk_size": min(batch_size, 50),  # 임베딩 포함 시 배치 크기 축소
                    "request_timeout": 300,             # 타임아웃 증가
                    "raise_on_error": False,
                    "raise_on_exception": False,
                    "max_retries": 3                    # 재시도 횟수 증가
                }
                
                # Ingest Pipeline이 지정된 경우 추가
                if pipeline:
                    for action in actions:
                        action["pipeline"] = pipeline
                
                try:
                    success_count, failed_items = bulk(**bulk_params)
                    total_indexed += success_count
                    
                    if failed_items:
                        failed_docs.extend(failed_items)
                        logging.warning(f"배치에서 {len(failed_items)}개 문서 인덱싱 실패")
                        
                        # 첫 번째 실패 원인만 로깅
                        if failed_items:
                            first_error = failed_items[0]
                            if 'index' in first_error and 'error' in first_error['index']:
                                error_info = first_error['index']['error']
                                logging.warning(f"실패 원인: {error_info.get('type', 'unknown')} - {error_info.get('reason', 'unknown')}")
                            
                except Exception as bulk_error:
                    logging.error(f"Bulk 인덱싱 중 예외 발생: {bulk_error}")
                    error_count += len(batch_data)
                    continue
                
                if show_progress:
                    progress = (i + batch_size) / len(concepts_data) * 100
                    logging.info(f"인덱싱 진행률: {progress:.1f}% ({total_indexed}/{len(concepts_data)})")
            
            # 인덱스 새로고침
            self.es.indices.refresh(index=self.index_name)
            
            logging.info(f"인덱싱 완료: 총 {total_indexed}개 문서 인덱싱")
            
            if failed_docs:
                logging.warning(f"실패한 문서 수: {len(failed_docs)}")
                # 심각한 실패가 많은 경우에만 파일 저장
                if len(failed_docs) > len(concepts_data) * 0.1:  # 10% 이상 실패시
                    with open(f"failed_indexing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
                        json.dump(failed_docs[:100], f, indent=2)  # 처음 100개만 저장
            
            # 성공한 문서가 80% 이상이면 성공으로 간주
            success_rate = total_indexed / len(concepts_data) if concepts_data else 1.0
            return success_rate >= 0.8
            
        except Exception as e:
            logging.error(f"인덱싱 중 오류 발생: {e}")
            return False
    
    def search_by_embedding(
        self,
        query_embedding: List[float],
        size: int = 10,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        임베딩을 사용한 유사도 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            size: 반환할 결과 수
            min_score: 최소 유사도 점수
            
        Returns:
            검색 결과 리스트
        """
        try:
            if not hasattr(self, "include_embeddings") or not self.include_embeddings:
                logging.warning("임베딩 비활성화 상태이므로 임베딩 검색을 건너뜁니다.")
                return []
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'concept_embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "min_score": min_score + 1.0,  # script_score는 1.0을 더함
                "size": size
            }
            
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "concept_id": hit["_source"]["concept_id"],
                    "concept_name": hit["_source"]["concept_name"],
                    "domain_id": hit["_source"]["domain_id"],
                    "vocabulary_id": hit["_source"]["vocabulary_id"],
                    "similarity_score": hit["_score"] - 1.0,  # 원래 점수로 복원
                    "full_data": hit["_source"]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"임베딩 검색 중 오류 발생: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 반환"""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            return {
                "document_count": stats["indices"][self.index_name]["total"]["docs"]["count"],
                "store_size": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "index_name": self.index_name
            }
        except Exception as e:
            logging.error(f"인덱스 통계 조회 실패: {e}")
            return {}
    
    def delete_index(self) -> bool:
        """인덱스 삭제"""
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logging.info(f"인덱스 삭제 완료: {self.index_name}")
                return True
            else:
                logging.info(f"삭제할 인덱스가 존재하지 않습니다: {self.index_name}")
                return True
        except Exception as e:
            logging.error(f"인덱스 삭제 실패: {e}")
            return False


if __name__ == "__main__":
    # 간단한 테스트
    logging.basicConfig(level=logging.INFO)
    indexer = ConceptElasticsearchIndexer(index_name="test_concepts")
    print("Elasticsearch 인덱서 초기화 완료")
    if indexer.es.ping():
        print("Elasticsearch 연결 테스트 성공")
    else:
        print("Elasticsearch 연결 실패")
