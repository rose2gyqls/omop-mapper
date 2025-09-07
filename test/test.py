from elasticsearch import Elasticsearch

# --- Elasticsearch 연결 정보 수정 ---
ES_HOST = "http://3.35.110.161:9200"
ES_USER = "elastic"
ES_PASSWORD = "snomed"
# ------------------------------------

try:
    # Elasticsearch 클라이언트 생성
    es = Elasticsearch(
        [ES_HOST],
        basic_auth=(ES_USER, ES_PASSWORD),
        request_timeout=30
    )

    # 클러스터의 모든 노드 정보 가져오기
    # 'settings'와 'roles' 정보만 필터링하여 요청
    nodes_info = es.nodes.info(metric=["settings", "roles"])

    print("--- Cluster Node Roles ---")

    ml_node_found = False
    for node_id, info in nodes_info["nodes"].items():
        node_name = info["name"]
        roles = info["roles"]
        
        print(f"Node Name: {node_name}")
        print(f"  - Roles: {roles}")
        
        if "ml" in roles:
            ml_node_found = True
            print("  - ✅ This node is an ML node.")
        else:
            print("  - ❌ This node is NOT an ML node.")
        print("-" * 20)

    if not ml_node_found:
        print("\n[CRITICAL] No ML nodes found in the cluster!")
        print("Please edit 'elasticsearch.yml' on at least one node to add the 'ml' role and restart it.")
    else:
        print("\n[INFO] ML node(s) found in the cluster.")

except Exception as e:
    print(f"An error occurred: {e}")