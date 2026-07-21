def test_health_discloses_synthetic_local_boundary(engine) -> None:
    from maoz_search.web import create_app

    client = create_app(engine).test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok", "synthetic_only": True, "network_inference": False}


def test_search_api_returns_no_raw_scores(engine) -> None:
    from maoz_search.web import create_app

    client = create_app(engine).test_client()
    response = client.post("/api/search", json={"query": "חינוך בלתי פורמלי", "filters": {}})
    assert response.status_code == 200
    result = response.json["results"][0]
    assert result["profile_id"] == "003SYN000000001"
    assert not {"dense_score", "lexical_score", "fusion_score"}.intersection(result)


def test_search_api_rejects_oversized_or_malformed_input(engine) -> None:
    from maoz_search.web import create_app

    client = create_app(engine).test_client()
    assert client.post("/api/search", data="not json", content_type="application/json").status_code == 400
    response = client.post("/api/search", json={"query": "א" * 201, "filters": {}})
    assert response.status_code == 400
