# 조형준 · Data Engineer

교통·공간 원천과 외부 API를 검증 가능한 데이터로 바꾸는 일을 합니다. 원천 보존, 정합성,
재실행과 장애복구를 중요하게 봅니다.

- 주 사용: Python, SQL, DuckDB, Parquet, PostgreSQL, Airflow, dbt, AWS
- 실무: TMAP 93,432,415행 감사, Legacy MDB 3,653,444행 변환, R→Python 49개 산출물 regression
- 관심 분야: Mobility Data, Data Quality, Batch Pipeline, Legacy Migration

## 대표 작업

| 프로젝트 | 해결한 문제 | 공개 근거 |
|---|---|---|
| [MobilityFlow](https://github.com/zodia8393/mobility-flow-platform) | 서울 교통 API 원천 보존, 품질검증, READY/BLOCKED 판정 | 115,570행 scheduled operation, 최근 10/10 PASS, 실행 영상 |
| [CatalogForge](https://github.com/zodia8393/catalog-forge) | 외부 페이지 수집의 retry, 중복, schema drift, 장애복구 | recovery rehearsal, benchmark, test |
| 개인 ETF 자동적립 시스템 | 외부 API 불명확 상태의 재요청 차단과 reconciliation | 계좌정보를 제외한 비식별 운영 사례 |

MobilityFlow의 PySpark는 `local[2]` 변환 검증 범위이며 AWS 배포는 Terraform 검증 범위입니다.
실제 AWS EC2·S3·systemd 운영 경험은 비공개 개인 자동화 시스템에서 수행했습니다.

Contact: [chohj_1019@naver.com](mailto:chohj_1019@naver.com)
