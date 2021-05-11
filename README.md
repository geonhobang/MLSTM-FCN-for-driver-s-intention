# MLSTM-FCN-for-driver's-intention
---
## 프로젝트 설명
1. 운전자의 의도를 예측할 수 있다면 사고를 예방 할 수 있지 않을까?하는 의문에서 시작된 프로젝트입니다.
일반적인 주행 상황보다 교차로 내에서 더 많은 사고가 발생하기에 내 차의 주행데이터, 운전자의 시선 데이터, 마지막으로 주변 차량과의 거리를 통해 현재 내 차량의 상태를 예측합니다.
2. 일반적으로 LSTM은 자연어 처리에 많이 이용됩니다. 하지만, 본 프로젝트에서는 현재로부터 3초전 데이터를 통해 차량의 상태를 예측하기 때문에 LSTM 모델을 이용하였습니다. 더욱 좋은 성능을 위해 FCN을 결합한 MLSTM-FCN모델을 이용하였습니다. 다음의 논문을 참고하였습니다.
 > Karim, F., Majumdar, S., Darabi, H., & Chen, S.
(2017). LSTM fully convolutional networks for time
series classification. IEEE access,6,
3. 해당 프로젝트에 사용한 모델을 통해 다른 상황에도 적용해 볼 수 있습니다. 예를들면, 주가와 같이 과거 데이터를 통해서 현재 혹은 미래를 예측하는 프로젝트에도 적용해 볼 수 있습니다.
---
## 선행조건
- Pytorch
- Sklearn(전처리 및 데이터 분할)
- visdom(트레인 및 테스트 상황 시각화)
- Pandas(커스텀 데이터 세트 로드)
---
## 기타사항
- 본 프로젝트에 사용된 데이터는 외부에서 구입한 데이터로 따로 배포할 수 없습니다.
- 모델의 Input Data는 각종 센서로부터 받아온 시계열 데이터가 Pandas와 Sklearn, torch.util.data.Dataload를 통해 (Batch size, Time step, Feature)로 변환됩니다.
---
## 개발인원
1. 방건호
2. 정주호
