# SKN10-3rd-3Team

# LLM Document Q&A System

# LLM 연동 내외부 문서 기반 질의 응답 시스템

> **문서 하나로 척척!**  
> 최신 LLM과 RAG 기술을 활용해 내외부의 다양한 문서를 기반으로 사용자 질의에 답하는 혁신적인 시스템입니다.

---

## 1. 주제 소개
이 프로젝트는 LLM(대규모 언어 모델)과 RAG(Retrieval-Augmented Generation) 기술을 연동하여, 회사 내부 문서부터 공개 자료까지 다양한 소스의 문서를 기반으로 사용자의 질문에 적절한 답변을 제공하는 시스템을 개발하는 것을 목표로 합니다.

## 2. 팀 소개
- **팀명:** [팀 이름]
- **팀원:** [팀원1, 팀원2, 팀원3, ...]  
  각 팀원은 데이터 수집, 전처리, 백엔드, 프론트엔드 등 다양한 역할을 맡아 프로젝트를 진행하고 있습니다.
- **프로젝트 목표:** 사용자에게 직관적이고 신뢰성 높은 질의 응답 서비스를 제공하여, 문서 기반 정보 검색 및 활용의 새로운 패러다임을 제시하는 것입니다.

## 3. 어떤 내외부 문서를 찾을지 내용 (공란)
- **내부 문서:**  
  - 회사 정책, 내부 보고서, 매뉴얼, 회의록 등 (예시)
- **외부 문서:**  
  - 뉴스, 논문, 기술 문서, 대중문화 자료 등 (예시)
  
*실제 적용 시, 프로젝트의 특성에 맞게 구체화 예정입니다.*

## 4. 문서 분야 선정한 이유
- **다양한 데이터 출처:** 내부와 외부의 다양한 문서를 대상으로 하여 정보의 폭과 깊이를 동시에 확보할 수 있음.
- **최신 정보 반영:** RAG를 통해 실시간 업데이트와 최신 정보 제공이 가능.
- **사용자 편의성:** 사용자가 필요한 정보를 빠르게 찾을 수 있도록 돕기 위해 선정.
- **다양한 적용 분야:** 기업 내부 관리, 고객 지원, 교육, 일반 대중 정보 제공 등으로 활용 범위가 넓음.

## 5. 문제 정의 및 해결방안 계획
- **문제 정의:**  
  - 기존 챗봇은 정적 데이터에 의존하거나, 특정 분야에 대한 심도 있는 정보를 제공하지 못함.
  - 최신 정보 업데이트 및 실시간 정보 반영에 한계가 있음.
- **해결방안:**  
  1. **문서 수집 및 전처리:** 다양한 내외부 문서를 자동으로 수집하고, 정제 및 분류.
  2. **임베딩 및 벡터 DB 구축:** 문서 데이터를 임베딩하여 유사도 기반 검색이 가능한 벡터 DB(예: FAISS, Milvus) 구축.
  3. **LLM 연동:** OpenAI API 혹은 오픈소스 LLM과 연동하여 검색된 문서를 기반으로 자연어 응답 생성.
  4. **실시간 데이터 연동 및 모니터링:** 외부 API 연동으로 최신 정보 반영 및 시스템 성능 모니터링.
- **목표:** 사용자에게 보다 정확하고 최신의 정보 기반 질의 응답 서비스를 제공하는 것.

## 6. 프로세스 흐름도
1. **문서 수집** → 2. **전처리** → 3. **임베딩 생성** → 4. **벡터 DB 저장** → 5. **사용자 질의 입력** → 6. **유사 문서 검색** → 7. **LLM 응답 생성** → 8. **사용자에게 출력**

*자세한 흐름도는 별도 다이어그램 파일(예: `process_flow.png`)로 첨부 예정입니다.*

## 7. 아키텍처
- **데이터 레이어:**  
  - 내외부 문서 및 외부 API를 통해 수집된 데이터를 저장하는 데이터베이스.
- **전처리 및 임베딩 레이어:**  
  - 문서 정제, 토큰화, 임베딩 생성 및 벡터 DB 구축.
- **서비스 레이어:**  
  - 사용자 질의 입력 처리, 벡터 DB 기반 유사 문서 검색, LLM 호출 및 응답 생성.
- **프론트엔드 & API 게이트웨이:**  
  - 사용자 인터페이스 및 RESTful API 제공.
- **실시간 데이터 연동 모듈:**  
  - 최신 정보 업데이트를 위한 외부 API 연동.

*아키텍처 다이어그램은 `architecture.png` 파일에 첨부 예정입니다.*

## 8. 시퀀스 다이어그램
1. **사용자**가 질의를 입력 →
2. **API 게이트웨이**가 요청 수신 →
3. **검색 모듈**이 벡터 DB 조회 →
4. **LLM 연동 모듈**이 관련 문서를 기반으로 응답 생성 →
5. **API 게이트웨이**가 응답 반환 →
6. **사용자**에게 최종 응답 출력

*자세한 시퀀스 다이어그램은 `sequence_diagram.png` 파일에 첨부 예정입니다.*

## 9. 일반 챗봇과 팀 프로젝트 챗봇의 차이점 및 성능 업 부분
- **일반 챗봇:**  
  - 정적 데이터에 기반한 일반 대화 지원.  
  - 실시간 정보 업데이트 및 문서 기반 심층 검색 기능 미흡.
- **팀 프로젝트 챗봇:**  
  - 내외부 문서를 기반으로 한 최신 정보 제공 및 심층 질의 응답 가능.
  - RAG 기술을 통해 임베딩 기반 유사도 검색 및 LLM 응답 생성.
  - 다양한 데이터 소스를 통합하여 사용자 맞춤형 답변 제공.
  - **성능 업:** 정보의 정확도, 최신성, 응답의 다양성과 깊이에서 탁월한 개선 효과.

## 10. 기대되는 효과
- **정보 접근성 향상:** 사용자들이 필요한 정보를 빠르고 정확하게 얻을 수 있음.
- **시간 및 비용 절감:** 복잡한 문서 검색 과정을 자동화하여 업무 효율 개선.
- **다양한 분야 응용:** 기업 내부 관리, 고객 지원, 교육, 대중 정보 제공 등 다방면에서 활용 가능.
- **사용자 만족도 증대:** 맞춤형 자연어 응답 제공으로 정보 탐색 경험을 대폭 향상.

---

*추가 내용 및 세부 사항은 팀 회의를 통해 점진적으로 보완할 예정입니다.*


