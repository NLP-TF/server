import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
import sys
import pytest
import pytest_asyncio
import asyncio
import torch
from httpx import AsyncClient
from app.db.database import engine, Base
import logging

logging.basicConfig(level=logging.ERROR)


@pytest_asyncio.fixture(autouse=True, scope="session")
async def create_test_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Optionally, drop tables after tests
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)


from pathlib import Path
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.schemas import UserType
from app.services.game_service import leaderboard as game_leaderboard


# Clear test data before each test
def clear_test_data():
    data_dir = Path("data")
    rankings_file = data_dir / "rankings.json"
    if rankings_file.exists():
        rankings_file.unlink()
    # Clear the in-memory leaderboard
    game_leaderboard.clear()


@pytest_asyncio.fixture(autouse=True)
async def run_around_tests():
    game_leaderboard.clear()
    clear_test_data()
    yield
    clear_test_data()
    game_leaderboard.clear()


@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def session_id(async_client):
    response = await async_client.post(
        "/api/v1/game/start", json={"nickname": "test_user", "user_type": "T"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    return data["session_id"]


@pytest.mark.asyncio
async def test_start_game(async_client):
    response = await async_client.post(
        "/api/v1/game/start", json={"nickname": "test_user", "user_type": "T"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data


@pytest.mark.asyncio
async def test_get_round(async_client):
    # 먼저 세션을 생성
    start_resp = await async_client.post(
        "/api/v1/game/start", json={"nickname": "test_user", "user_type": "T"}
    )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    # session_id와 round_number로 라운드 정보 요청
    response = await async_client.get(f"/api/v1/game/round/{session_id}/1")
    assert response.status_code == 200
    data = response.json()
    assert "situation" in data
    assert "example_response" in data
    assert data["round_number"] == 1


@pytest.mark.asyncio
async def test_submit_response(async_client, session_id):
    response = await async_client.post(
        "/api/v1/game/submit",
        json={
            "session_id": session_id,
            "round_number": 1,
            "user_response": "This is a test response.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "score" in data


@pytest.mark.asyncio
async def test_get_summary(async_client, session_id):
    await async_client.post(
        "/api/v1/game/submit",
        json={
            "session_id": session_id,
            "round_number": 1,
            "user_response": "This is a test response.",
        },
    )
    response = await async_client.get(f"/api/v1/game/summary/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "nickname" in data
    assert "total_score" in data
    assert "round_scores" in data
    assert isinstance(data["round_scores"], list)
    assert len(data["round_scores"]) > 0
    round_score = data["round_scores"][0]
    assert "round_number" in round_score
    assert "score" in round_score
    assert "user_response" in round_score
    assert "is_correct_style" in round_score
    assert "percentile" in data
    assert "rank" in data
    assert "top_players" in data
    assert isinstance(data["top_players"], list)
    assert "feedback" in data


@pytest.mark.asyncio
async def test_get_rankings(async_client, monkeypatch):
    # Mock prediction functions
    def mock_predict_tf_style(text):
        try:
            if "Test response" in text:
                player_num = int(text.split()[-1].rstrip("."))
                score = 0.7 + (player_num * 0.1)
                f_prob = min(score, 0.9)
                result = {"F_prob": f_prob * 100, "T_prob": (1 - f_prob) * 100}
            else:
                result = {"F_prob": 50.0, "T_prob": 50.0}
            return result
        except Exception:
            return {"F_prob": 50.0, "T_prob": 50.0}

    monkeypatch.setattr(
        "app.services.game_service.predict_tf_style", mock_predict_tf_style
    )
    monkeypatch.setattr("app.models.predict.predict_tf_style", mock_predict_tf_style)

    # Mock model/tokenizer
    def mock_get_model():
        class MockModel:
            def __init__(self):
                self.config = type("", (), {"num_labels": 2})()

            def __call__(self, *args, **kwargs):
                import torch

                return type("", (), {"logits": torch.tensor([[0.0, 0.0]])})

            def eval(self):
                return self

            def to(self, device):
                return self

            @property
            def parameters(self):
                import torch

                return [torch.tensor([0.0])]

        return MockModel()

    def mock_get_tokenizer():
        class MockTokenizer:
            def __init__(self):
                pass

            def __call__(self, *args, **kwargs):
                import torch

                return {
                    "input_ids": torch.tensor([[0]]),
                    "attention_mask": torch.tensor([[1]]),
                }

        return MockTokenizer()

    monkeypatch.setattr("app.models.load_model.get_model", mock_get_model)
    monkeypatch.setattr("app.models.load_model.get_tokenizer", mock_get_tokenizer)

    # Setup test data dir and rankings file
    from pathlib import Path

    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True, parents=True)
    test_rankings_file = test_data_dir / "rankings.json"
    if test_rankings_file.exists():
        test_rankings_file.unlink()

    # Set env and reload services/singleton
    monkeypatch.setenv("RANKING_DATA_DIR", str(test_data_dir.absolute()))
    from importlib import reload
    from app.services import ranking_service as ranking_service_module

    reload(ranking_service_module)
    from app.services.ranking_service import RankingService

    RankingService._instance = None
    ranking_service = RankingService()
    from app.services import game_service

    reload(game_service)
    game_service.ranking_service = ranking_service

    # Test logic
    initial_rankings = ranking_service.get_rankings()
    assert initial_rankings["total_players"] == 0
    test_players = ["player1", "player2", "player3"]
    for i, nickname in enumerate(test_players, 1):
        response = await async_client.post(
            "/api/v1/game/start", json={"nickname": nickname, "user_type": "T"}
        )
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        response_text = f"Test response {i}"
        response = await async_client.post(
            "/api/v1/game/submit",
            json={
                "session_id": session_id,
                "round_number": 1,
                "user_response": response_text,
            },
        )
        assert response.status_code == 200
        response = await async_client.get(f"/api/v1/game/summary/{session_id}")
        assert response.status_code == 200
        current_rankings = ranking_service.get_rankings()

    # 강제로 랭킹 파일 저장
    await ranking_service._save_rankings()

    # 디버깅 파일 생성
    import os

    with open("rankings_debug_test.txt", "w") as f:
        f.write(
            f"ranking_service.rankings_file: {getattr(ranking_service, 'rankings_file', 'N/A')}\n"
        )
        f.write(f"RANKING_DATA_DIR: {os.environ.get('RANKING_DATA_DIR')}\n")
        f.write(f"test_rankings_file: {test_rankings_file}\n")
        f.write(
            f"rankings_file.exists: {getattr(ranking_service, 'rankings_file', None) and Path(ranking_service.rankings_file).exists()}\n"
        )
        f.write(f"test_rankings_file.exists: {test_rankings_file.exists()}\n")

    assert test_rankings_file.exists()


if __name__ == "__main__":
    import asyncio

    asyncio.run(pytest.main(["-v"]))
