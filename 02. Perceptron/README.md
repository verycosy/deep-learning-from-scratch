# Perceptron

- 신경망(딥러닝)의 기원이 되는 알고리즘
- 퍼셉트론은 다수의 신호를 입력으로 받아 하나의 신호를 출력 = (앞으로 전달)
- 복수의 입력 신호 각각에 고유한 가중치를 부여하는데, 가중치는 각 신호가 결과에 주는 영향력을 조절하는 요소로 작용.
- 퍼셉트론의 매개변수 값을 정하는 것은 컴퓨터가 아니라 인간.
- 인간이 직접 진리표라는 '학습 데이터'를 보면서 매개변수의 값을 생각함.
- 기계학습 문제는 이 매개변수의 값을 정하는 작업을  컴퓨터가 자동으로 하도록 만듦.
- '학습'이란 적절한 매개변수 값을 정하는 작업
- 사람은 퍼셉트론의 구조(모델)를 고민하고 컴퓨터에 학습할 데이터를 주는 일을 한다.
- 퍼셉트론의 구조는 AND, NAND, OR 게이트 모두에서 같다. 매개변수(가중치와 임계값)이 다를 뿐.
- **w는 각 입력 신호가 결과에 주는 영향력(중요도)을 조절하는 매개변수**
- **b는 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력)하느냐를 조정하는 매개변수**