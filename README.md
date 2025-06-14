# 너… T야? 🤖💬

**MBTI 기반 T/F 위로 방식 분류 게임 – FastAPI + KoBERT + Hugging Face**

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

## 🚀 API 엔드포인트

### 1. 게임 시작
```
POST /game/start
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
GET /game/round/{round_number}
```

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
POST /game/score
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
GET /game/summary/{session_id}
```

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
