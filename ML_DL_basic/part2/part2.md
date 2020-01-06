## 파이썬을 활용한 정형데이터 분석과 시각화

#### 1. 다양한 데이터 타입 & 데이터 수집

#### 2. 데이터 분석 & 데이터 시각화 (파이썬)





### 1. 다양한 데이터 타입 & 데이터 수집

**데이터 타입**

- 정형 데이터 (Structured data)
  - 관계형 데이터베이스
  - Spread sheets
- 반정형 데이터 (Semi-structured data)
  - System log
  - Sensor data
  - HTML
- 비정형 데이터 (Unstructured data)
  - 이미지 / 비디오
  - 소리
  - 문서



**데이터 수집**

- 다양한 tools
  - Google Analaytics
  - Elastic Stack or ELK Stack
  - Zeppelin
- API & Web Scraper
- 공공데이터 & Open data (APIs & files)
  - [공공 데이터 포털](https://www.data.go.kr/)
  - [국가 통계 포털](http://kosis.kr/index/index.do)
  - [MDIS](https://mdis.kostat.go.kr/index.do)
- 기타 Datasets
  - [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)
  - [Google AI Datasets](https://ai.google/)
  - [Google Dataset Search](https://toolbox.google.com/datasetsearch)
  - [SKT BigData Hub](https://www.bigdatahub.co.kr/index.do)
  - [Kaggle competition datasets](https://www.kaggle.com/datasets)



### 2. 데이터 분석 & 데이터 시각화 (파이썬)

1. 데이터 입력 및 데이터 전처리
2. 데이터 탐색 (Data Exploration)
   - 데이터 정렬, 데이터 정규화, 순위 비교,,,
3. 데이터 시각화 (Data Visualization)
   - 지도 시각화 : Folium Library
   - 지도 데이터 : GeoJSON

**Pandas Funcions**

- pd.read_excel()
- df['열이름'].apply(함수) : DF의 해당 열의 각 데이터에 특정 함수를 적용
- pd.pivot_table(df, index='열 이름', aggfunc = 함수명) : 원본 DF에서 특정 열 기준으로 새로운 DF를 만듦
- df.drop(['행이름'])
- df.rename(columns= {'기존 열 이름' : '새로운 열 이름'}, inplace = True)
- df.sort_values(by= '열 이름', ascending=False,inplace=True) : 정렬
- df_1.join(df_2): df_1의 index 기준으로 df_2의 index 중 매칭되는 데이터 merge



## +Appendix

**Numpy**

- Computational Science 분야에 자주 활용되는 Python 라이브러리
- 처리 속도가 빠른 다차원 array 객체와 이를 다룰 수 있는 다양한 함수들을 제공
- 상당 부분의 내부 코드가 C나 Fortran으로 작성되어 있어 실행 속도를 빠르게 끌어올림



**Panda**

- 정형 데이터의 전처리와 각종 연산을 효과적으로 할 수 있도록 도와주는 Python 라이브러리
- 엑셀의 sheet와 유사항 형태의 DataFrame을 활용해 데이터 쉽게 처리
- Numpy에 기반을 두고 있음



**Matplotlib**

- Python의 대표적인 데이터 & 그래프 시각화 라이브러리
- 선 그래프, 히스토그램, 산점도 등 각종 시각화 방식을 지원