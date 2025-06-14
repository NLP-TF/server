# 너… T야? 🤖💬

**MBTI 기반 T/F 위로 방식 분류 게임 – FastAPI + KoBERT + Hugging Face**

---

## 🚀 빠른 시작

### 1. 가상환경 설정 및 의존성 설치

```bash
# 가상환경 생성 (macOS/Linux)
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 추가 의존성 설치 (만약 requirements.txt에 없을 경우)
pip install torch transformers peft
```

### 2. 서버 실행

```bash
# 개발 모드로 실행 (자동 리로드 활성화)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
```

### 3. API 테스트

#### 게임 시작
```bash
curl -X POST http://127.0.0.1:8000/game/start \
  -H "Content-Type: application/json" \
  -d '{"nickname":"테스트유저", "user_type":"T"}'
```

#### 라운드 정보 조회
```bash
# 1라운드 정보 조회
curl http://127.0.0.1:8000/game/round/1
```

#### 응답 제출 및 점수 확인
```bash
# 세션 ID는 게임 시작 시 받은 값을 사용하세요
curl -X POST http://127.0.0.1:8000/game/score \
  -H "Content-Type: application/json" \
  -d '{"session_id":"세션_ID", "user_response":"괜찮아? 다음에는 더 잘할 수 있을 거야!", "round_number":1}'
```

#### 게임 결과 요약
```bash
# 세션 ID로 결과 조회
curl http://127.0.0.1:8000/game/summary/세션_ID
```

#### T/F 분류 모델 직접 테스트
```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "가서 친구한테 얘기해봐."}'
```

### 4. 테스트

```bash
# 유닛 테스트 실행
pytest

# 코드 커버리지 확인 (pytest-cov 설치 필요)
pytest --cov=app tests/
```

---

## 🎮 프로젝트 소개

**'너T야?'**는 MBTI의 T(Thinking)형과 F(Feeling)형의 위로 스타일 차이를 AI가 학습하고, 사용자의 문장을 평가하여 점수화하는 인터랙티브 게임입니다.  
사용자는 자신과 반대 성향의 위로 방식으로 문장을 작성하고, AI는 KoBERT 기반 분류 모델로 해당 문장의 'T/F스러움'을 수치화해 피드백합니다.

---

## 🧠 주요 기능

- 사용자 입력 문장을 기반으로 T/F 유사도 예측
- 점수화 및 누적 점수 관리
- 라운드별 상황 제공 및 분석 결과 리턴
- 유저 순위 및 백분위 통계 제공

## 🛠 문제 해결

### 모델 로딩 문제
- 모델이 로드되지 않을 경우 `model_cache` 디렉토리를 삭제하고 재시도하세요:
  ```bash
  rm -rf model_cache
  ```

### 토큰 관련 오류
- `token_type_ids` 관련 오류가 발생하면 서버를 재시작하세요.

### 로그 확인
- 서버 로그는 터미널에 실시간으로 출력됩니다.
- `--log-level debug` 옵션으로 상세한 로그를 확인할 수 있습니다.

---

## 🚀 API 엔드포인트

### 1. 게임 시작
```
POST /api/v1/game/start
```

**요청 본문 (JSON):**
```json
{
  "nickname": "사용자닉네임",
  "user_type": "T"  // 또는 "F"
}
```

**성공 응답 (200):**
```json
{
  "session_id": "생성된_세션_ID",
  "message": "Game started successfully"
}
```

### 2. 라운드 정보 조회
```
GET /api/v1/game/round/{round_number}
```

**경로 파라미터:**
- `round_number`: 라운드 번호 (1-5)

**성공 응답 (200):**
```json
{
  "round_number": 1,
  "situation": "친구가 시험에 떨어졌을 때",
  "example_response": "너무 속상하겠다. 괜찮아? 기분이 어때?"
}
```

### 3. 응답 제출 및 점수 획득
```
POST /api/v1/game/score
```

**요청 본문 (JSON):**
```json
{
  "session_id": "세션_ID",
  "user_response": "사용자_응답_텍스트",
  "round_number": 1
}
```

**성공 응답 (200):**
```json
{
  "score": 78.5,
  "message": "Response scored successfully"
}
```

### 4. 게임 결과 요약
```
GET /api/v1/game/summary/{session_id}
```

**경로 파라미터:**
- `session_id`: 게임 세션 ID

**성공 응답 (200):**
```json
{
  "session_id": "세션_ID",
  "nickname": "사용자닉네임",
  "user_type": "T",
  "total_score": 350.5,
  "round_scores": [
    {
      "round_number": 1,
      "score": 78.5,
      "user_response": "사용자_응답_텍스트",
      "is_correct_style": true
    }
  ],
  "percentile": 85.5,
  "rank": 4,
  "top_players": [
    {"nickname": "최고수", "user_type": "F", "total_score": 480, "timestamp": "2025-06-14T12:00:00"},
    {"nickname": "중간자", "user_type": "T", "total_score": 420, "timestamp": "2025-06-14T11:30:00"},
    {"nickname": "초보자", "user_type": "F", "total_score": 380, "timestamp": "2025-06-14T10:45:00"}
  ],
  "feedback": "훌륭해요! 감정형(F) 스타일을 매우 잘 이해하고 계시네요!"
}
```

## 🎮 게임 진행 흐름

1. 사용자가 닉네임과 본인의 MBTI 유형(T/F)을 입력하여 게임을 시작합니다.
2. 각 라운드마다 특정 상황이 제시됩니다.
3. 사용자는 제시된 상황에 대해 반대 성향의 스타일로 응답을 작성합니다.
   - T 유형 사용자: F 스타일로 감정을 표현하는 응답 작성
   - F 유형 사용자: T 스타일로 논리적인 응답 작성
4. AI 모델이 응답을 분석하여 점수를 매기고 피드백을 제공합니다.
5. 5라운드가 종료되면 총점, 순위, 백분위, 상위 플레이어 정보 등을 포함한 종합 결과를 확인할 수 있습니다.

## 📌 커밋 컨벤션 규칙

커밋 메시지는 **"기능: 기능설명"** 형식으로 작성합니다.

### ✅ 커밋 유형

| 유형       | 설명                                               |
| ---------- | -------------------------------------------------- |
| `feat`     | 새로운 기능 추가 또는 기존 기능 수정               |
| `fix`      | 기능에 대한 버그 수정                              |
| `build`    | 빌드 관련 수정                                     |
| `chore`    | 패키지 매니저 수정 및 기타 수정 (예: `.gitignore`) |
| `docs`     | 문서(주석) 수정                                    |
| `style`    | 코드 스타일 및 포맷팅 수정 (기능 변경 없음)        |
| `refactor` | 기능 변경 없이 코드 리팩터링 (예: 변수명 변경)     |

### 📝 커밋 메시지 예시

```bash
feat: Add user authentication
fix: Resolve login button bug
```
