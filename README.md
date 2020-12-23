# Dandelion counter 알고리즘 설명서

연규민

## dependency

- opencv-python
- numpy

## 알고리즘 설명

키워드

RGB to HSV, opening, dfs, minimum bounding box

- 이미지를 리사이징 합니다.
- RGB color space image를 HSV color space image로 변환 시킵니다.
- HSV color space image에서 커스텀으로 설정한 yellow 픽셀범위의 픽셀들을 추출합니다.
- 노이즈 제거를 위해서 2x2, 3x3 opening을 시행합니다.(`_cancel_noise_with_opening()`)
- yellow 픽셀들의 클러스터를 추출합니다.(`_get_number_of_cluster_and_cluster_info_list()`)
  - 여기서 클러스터는 상하좌우가 연결된 픽셀들의 집합을 의미합니다.
  - 탐색은 dfs를 사용합니다.(`_dfs()`)
  - 단순히 클러스터의 개수로 답을 내면 단일 클러스터에서 민들레가 겹치는 경우를 전부 1개로 취급하게 되므로, overlapped cluster를 파악하는 전략이 필요합니다.
- 이미지에서 추출된 클러스터 중에서 overlapped cluster를 구합니다.
  - overlap의 판단 기준은 다음과 같습니다. (`_is_overlapped_cluster()`)
    - 1# overlapped 클러스터인지 판정하고 싶은 클러스터가 포함하는 픽셀개수가, 이미지 전체 cluster들의 cluster가 포함하는 픽셀개수의 중앙값의 2배와, 100픽셀 중에서 더 큰 수 보다 크다면 overlap이라고 판단합니다.
      - 즉, 크기가 비정상적으로 큰 클러스터의 경우, overlapped으로 판단
    - 2# 아래의 두가지 경우를 만족하면 overlap으로 판단합니다.
      - overlapped 클러스터인지 판정하고 싶은 클러스터가 포함하는 픽셀개수가, 이미지 전체 cluster 들의 pixel number의 median보다 1.6배(파라미터) 보다 많은 경우
      - overlapped 클러스터인지 판정하고 싶은 클러스터가 포함하는 픽셀개수가, 해당 클러스터의 픽셀의 minimum bounding box의 넓이의 0.7배 보다 작은 경우
        - 일반적인 overlapped이 아닌 민들레의 경우, minimum bounding box와 cluster픽셀 개수의 차가 그다지 크지 않음에 착안
- overlapped cluster가 실제로 몇개가 겹쳐있는지 구합니다.
  - 겹쳐있는 개수 판단 기준은 다음과 같습니다.
    - overlapped 로 판정된 클러스터의 픽셀 개수를 이미지 전체 cluster 들의 pixel number의 median과 50 pixel 중 더 큰 값으로 나누고 소수 첫쨰자리로 반올림을 한 값을 겹쳐있는 개수로 판단합니다.
    - 단, overlapped cluster는 무조건 겹쳐있음이 보장되므로 개수는 2이상이어야 합니다.
- overlapped cluster와 겹친 개수를 고려한 조정답을 구합니다.
- 답을 출력합니다.

## 알고리즘의 동작 결과를 이미지를 가지고 눈으로 확인하는 법

- `if __name__ == '__main__'` 아래의 #for check로 되어있는 부분 아래를 주석 해제합니다.
- `file_path`를 분석을 원하는 파일 패스로 설정합니다.
- `dandelion_counter.get_dandelion_number(show_picture=True)`인것을 확인하고 실행하면, 눈으로 직접 알고리즘이 어떤 영역을 민들레로 판단했고, 해당 영역에 민들레가 몇개가 들어있는지 판단한 것을 숫자로 나타낸 이미지를 확인할 수 있습니다.
