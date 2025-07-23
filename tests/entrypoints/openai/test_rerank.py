# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests

from vllm.entrypoints.openai.protocol import RerankResponse

from ...utils import RemoteOpenAIServer

MODEL_NAME = "BAAI/bge-reranker-base"
DTYPE = "bfloat16"


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


@pytest.fixture(scope="module")
def server():
    args = ["--enforce-eager", "--max-model-len", "100", "--dtype", DTYPE]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_texts(server: RemoteOpenAIServer, model_name: str):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].relevance_score >= 0.9
    assert rerank.results[1].relevance_score <= 0.01


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_top_n(server: RemoteOpenAIServer, model_name: str):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.", "Cross-encoder models are neat"
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                        "top_n": 2
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].relevance_score >= 0.9
    assert rerank.results[1].relevance_score <= 0.01


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_max_model_len(server: RemoteOpenAIServer, model_name: str):

    query = "What is the capital of France?" * 100
    documents = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents
                                    })
    assert rerank_response.status_code == 400
    # Assert just a small fragments of the response
    assert "Please reduce the length of the input." in \
        rerank_response.text


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_max_tokens_per_doc(server: RemoteOpenAIServer, model_name: str):
    query = "programming languages"
    documents = [
        "Python is a high-level programming language known for its simplicity and readability",
        "JavaScript is a versatile programming language primarily used for web development",
        "Rust is a systems programming language focused on safety and performance"
    ]
    
    response_full = requests.post(server.url_for("rerank"), json={
        "model": model_name,
        "query": query,
        "documents": documents,
    })
    response_full.raise_for_status()
    
    response_truncated = requests.post(server.url_for("rerank"), json={
        "model": model_name,
        "query": query,
        "documents": documents,
        "max_tokens_per_doc": 4
    })
    response_truncated.raise_for_status()
    
    full_results = response_full.json()["results"]
    truncated_results = response_truncated.json()["results"]
    
    for full_doc, truncated_doc in zip(full_results, truncated_results):
        assert len(truncated_doc["document"]["text"]) < len(full_doc["document"]["text"])
    
    assert len(truncated_results) == 3


def test_invocations(server: RemoteOpenAIServer):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    request_args = {
        "model": MODEL_NAME,
        "query": query,
        "documents": documents,
    }

    rerank_response = requests.post(server.url_for("rerank"),
                                    json=request_args)
    rerank_response.raise_for_status()

    invocation_response = requests.post(server.url_for("invocations"),
                                        json=request_args)
    invocation_response.raise_for_status()

    rerank_output = rerank_response.json()
    invocation_output = invocation_response.json()

    assert rerank_output.keys() == invocation_output.keys()
    for rerank_result, invocations_result in zip(rerank_output["results"],
                                                 invocation_output["results"]):
        assert rerank_result.keys() == invocations_result.keys()
        assert rerank_result["relevance_score"] == pytest.approx(
            invocations_result["relevance_score"], rel=0.01)
