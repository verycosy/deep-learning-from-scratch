# Neural Network

- 신경망은 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습한다. = (사람이 수동으로 할 필요 없음)
- **(단순) 퍼셉트론**은 단충 네트워크에서 계단 함수를 활성화 함수로 사용한 모델을 가리킴.
- **다층 퍼셉트론**은 신경망(여러 층으로 구성되고 시그모이드 함수, ReLu 함수 등의 매끈한 활성화 함수를 사용하는 네트워크)을 가리킴.
- 계단 함수 : 임계값을 경계로 출력이 바뀌는 함수 (비선형)
- 신경망에서는 활성화 함수로 비선형 함수를 사용해야 한다.   
- 선형 함수의 문제는 층을 아무리 깊게 해도 '은닉층이 없는 네트워크'로도 똑같은 기능을 할 수 있다는 것.  
ex) h(x) = cx 일 때, 3층 네트워크라면 y(x) = h(h(h(x))) = ccc*x = ax  
따라서 층을 쌓는 혜택을 얻고 싶다면 반드시 비선형 함수를 사용해야 한다.
- 머신러닝과 마찬가지로 신경망도 두 단계를 거쳐 문제를 해결한다. 훈련(학습) 데이터를 사용해 가중치 매개변수를 학습하고, 추론 단계에서는 앞서 학습한 매개변수를 사용하여 입력 데이터를 분류한다.
- 추론 과정 = 순전파(Forward Propagation)
- 파이썬에는 pickle이라는 편리한 기능이 있다. 프로그램 실행 중에 특정 객체를 파일로 저장하는 기능. 저장해둔 pickle 파일을 로드하면 실행 당시의 객체를 즉시 복원할 수 있음. 덕분에 MNIST 데이터를 순식간에 준비 가능.
- 정규화(Normalization) : 데이터를 특정 범위로 변환하는 처리
- 전처리(Pre-processing) : 신경망의 입력 데이터에 특정 변환을 가하는 것
- 배치(batch) : 하나로 묶은 입력 데이터
- 수치 계산 라이브러리 대부분이 큰 배열을 효율적으로 처리할 수 있도록 고도로 최적화되어 있기 때문에 배치 처리를 해주는 것이 좋다. 그리고 커다란 신경망에서는 데이터 전송이 병목으로 작용하는 경우가 자주 있는데, 배치 처리를 함으로써 버스에 주는 부하를 줄일 수 있다(느린 I/O를 통해 데이터를 읽는 횟수가 줄어, 빠른 CPU나 GPU로 순수 계산을 수행하는 비율이 높아짐). 즉, 컴퓨터에서는 큰 배열을 한꺼번에 계산하는 것이 분할된 작은 배열을 여러 번 계산하는 것보다 빠르다.