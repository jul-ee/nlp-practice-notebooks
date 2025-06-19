# 📋 Reuters News Classification Project

46개 카테고리로 라벨링된 로이터 뉴스 기사에 대해 

TF-IDF 기반 전통 ML과 Word2Vec 기반 DL을 비교‧분석하여, 주제 분류에 가장 효율적인 파이프라인을 찾는 것을 목표로 합니다.

> 🔗 nlp-practice-notebooks > [`06_reuters_news_classification_project.ipynb`](https://github.com/jul-ee/nlp-practice-notebooks/blob/main/nlp_practice_notebooks/06_reuters_news_classification_project.ipynb)

<br>
<br>

## 프로젝트 목표

짧은 뉴스 기사에 대한 주제 분류 작업에서 “가벼운 전통 ML 파이프라인”과 “딥러닝 기반 시퀀스 모델” 중 어떤 접근이 더 효과적인지 실험적으로 검증

1. 어휘 집합(단어장) 크기가 전통적인 TF-IDF + ML 모델의 성능에 미치는 영향
2. 벡터화 방식(TF-IDF vs. Word2Vec) 변화가 ML / DL 모델 간 상대적 성능 차이에 미치는 영향


<br>
<br>

## 사용 데이터셋

- 출처: kears.datasets의 reuters 데이터
- 문서 수: 11,228 건 (학습 8,982 / 테스트 2,246)
- 라벨링: 46개 경제·무역·원자재 뉴스 주제
- 특징
  - 짧은 기사: 평균 ≈ 120 tokens
  - 다중 클래스 & 불균형


<br>
<br>

## 실험 설계

평가지표: Accuracy, Weighted f1-score

| 실험 | 변수 | 비교 모델 |
| --- | --- | --- |
| **① 어휘 집합 크기** | ‣ `Voca_size ∈ {500, 5 000, 10 000, inf}`<br><br>‣ 벡터화: TF-IDF | NB, Complement NB, Logistic Regression, Linear SVM, Decision Tree, Random Forest, Gradient Boosting, Voting |
| **② 벡터화 방식** | `DTM(TF-IDF)` vs `Word2Vec 평균` | ‣ ML: XGBoost<br>‣ DL: Dense DNN, RNN(10·20 epoch) |


<br>
<br>

## 결과 요약

| 최고 성능 조합 | Accuracy | F1-Score |
| --- | --- | --- |
| **TF-IDF 10 k + Logistic Regression** | **0.8110** | **0.8055** |
| Word2Vec + RNN (20 epoch) | 0.7689 | 0.7559 |
| Word2Vec + XGBoost | 0.7306 | 0.7133 |

> ▲ 전체 결과 테이블은 노트북 참조

<br>
<br>

## 주요 인사이트

로이터 뉴스 분류처럼 문서가 비교적 짧고 주제어가 뚜렷한 문제에서는 단순 TF-IDF + 선형 분류기가 가장 높은 성능을 보인다는 것을 알 수 있었음.

희소·고차원 벡터에서 분할 규칙 생성이 어려워 Decision Tree, Random Forest가 0.62 ~ 0.70 수준에 그쳤고, Gradient Boosting(0.77)도 선형 모델을 넘지 못하였음.

상위 약 5 000 ~ 10 000개의 핵심 토큰만으로도 카테고리를 충분히 구분할 수 있고, 단어장을 무한대로 확장하면 희귀어 노이즈가 늘어나 오히려 성능이 떨어진다는 점도 확인할 수 있었음.

동일한 데이터 규모와 문서 길이 조건에서 RNN이 순서 정보를 반영**하고도 Logistic Regression을 넘지 못한 것을 통해 별도의 최적화 없이는 학습 효율, 일반화 측면에서 선형 모델에 비해 불리하다는 사실을 확인함.

모델을 복잡하게 만들기 전에, 데이터 특성과 단순 모델로도 충분한지 먼저 점검하는 과정이 중요하다는 것을 인지함.


<br>

>본 프로젝트는 데이터·자원 제약 환경에서 재현성 높고 비용 대비 효율적인 분류 파이프라인 비교 실험에 중점을 두었습니다.
