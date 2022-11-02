1.SQL 연산순서
From
where
Group by
Having
Select
Order by

DML
Select insert update delete 
DDL
alter create modify drop
TCL
rollback commit
DCL
grant revoke

2.Distinct
어떤 컬럼값들의 중복을 제거한 결과를 출력한다
select distinct col from table;
select distinct col1,col2 from table; 의 경우에는 col1과 col2의 값이 모두 같지 않은거만 출력한다 (주의필요)

3.Alias
select 절에서 사용가능,where절에서는 사용불가
select col as name from table; = select col name from table;

4.concat
select col1+col2+col3 from table (SQL 서버)
select col1 || col2 || col3 from table (oracle 버전)
select concat(col1,col2) from table; (기억해야할것! 연산자가 2개!)

5.논리연산자
Not ~가 아니다
AND A 그리고 B (둘다만족)
OR A 또는 B (둘중 하나만 만족)

6.SQL 연산자
A BETWEEN B AND C (B<=A<=C)
A IN (1,2,3) A=1 OR A=2 OR A=3
A LIKE '_BLE*' A의 값중 2,3,4번째 값이 BLE인 모든 데이터 출력

7.ESCAPE
AND EMAIL LIKE '@_%'ESCAPE '@'
아무문자에서나 사용가능하다

8.ROWNUM,TOP
ORACLE에서는 WHERE절 옆에 ROWNUM
SQL 서버의 경우 SELECT 옆에 TOP

9.NULL의 정의
모르는 값,정의되지 않은값 (공백이나 0과는 다름)
산술연산에서 NULL이 들어가면 NULL이 출력된다

조건절에서 NULL이 들어가면 FALSE 를 반환한다

집계함수 (SUM,COUNT,MIN,MAX ETC)에서 NULL은 데이터 대상에서 제외된다
정렬시에는 오라클에서는 가장 큰것으로 분류되고 SQL서버에서는 가장 작은값으로 분류된다

NVL(COL,0) COL이 널이면 0 아니면 COL
NVL2(COL1,0) COL이 널이면 0 아니면 1
ISNULL(COL,0) COL이 널이면 0 아니면 COL
NULLIF(COL,0) COL이 0이면 널 아니면 COL
COALESCE(COL1,COL2,COL3 ~) NULL이 아닌 첫번째 값 반환


10.정렬
느려질수있음
가장 마지막에 실행해야함
null이 어디에 오는지 알수있음

컬럼명으로 정렬, 앞의 기준이 같을때는 그 다음 컬럼으로 정렬한다
기본값은 오름차순은 asc 내림차순은 desc
order by col1,col2 desc
출력순서(번호)로 정렬, select 절의 출력순서로 정렬순서를 지정한다
order by 2,1 desc

11.숫자함수
round(222.45,1)소수점 둘째자리에서 반올림하여 첫째자리까지 출력한다
round(222.45,0)소수점 첫째자리에서 반올림하여 정수만 출력한다
-1파라미터는 1의 자리에서 반올림하여 정수를 출력한다

올림함수
oracle버전 ceil
SQL 서버 ceiling 
파라미터 사용법은 round 와 같다

버림함수
floor 파라미터 사용법은 round와 같음

12.문자함수
lower,upper 소문자로, 대문자로
trim,ltrim,rtrim 양쪽공백제거,왼쪽,오른쪽 공백제거함수
lpad,rpad 특정자리를 정하고 왼쪽/오른쪽의 공백을 채워주는 함수
ex) select lpad('A',5,'*')from dual;
->***A rpad이면 A***
substr select substr('korea',2,2)from dual ; -> or 이 출력된다
instr select instr('CORPORATE FLOOR','PO')as idx from dual; -> 4가 출력된다

13.날짜함수
to_char  (날짜형 데이터를 문자로 출력한다)
select to_char(sysdate,'YYYY-MM-DD')from dual;

to_date (문자형 데이터를 날짜형으로 출력)
select to_date('2022-09-22')from dual;

oracle버전 ->sysdate
SQL 서버버전 -> getdate()

13.조건문
decode
select decode(col1,'a',1,'b',2,3)from dual;
col이 a면 1  b면 2 아니면 3

case
case when col='A' then 1
    when col='B' then 2
    else 3 end;

case col when 'A' then 1
    when 'B' then 2
    else 3 end;

decode 문과 같은 기능을한다

14.집계함수
count,min,sum,max 등 
집계시 NULL은 포함되지않는다
(1,NULL,2,3,NULL)의 데이터를 기준으로할때
COUNT() = 3
SUM=()6
AVG=()2
MIN=()1
MAX=()3

COL1=NULL,NULL,1
COL2=2,3,2
COL3=1,NULL,NULL

SELECT SUM(COL1+COL2+COL3)FROM DUAL;
여기에서 먼저 SUM을 생각하지말고 COL1+COL2+COL3를 먼저 생각해보면 첫번째행은 NULL+NULL+1이기에 NULL이 반환되고 마지막 세번째 행도 마찬가지
따라서 두번째행의 2+3+2의 값인 7이 결과값으로 출력된다

반대로 SUM(COL1)+SUM(COL2)+SUM(COL3)의 값은 3+3+3이므로 9가 출력됨
차이를 아는것이 중요하다

15.GROUP BY
집약기능을 가지고있다 (다수의 행을 하나로 합침)
GROUP BY 절에 온 컬럼만 SELECT 절에 올수있다

16.JOIN
NATURAL JOIN
반드시 두 테이블 간의 동일한 이름,타입을 가진 컬럼이 필요
조인에 이용되는 컬럼은 명시되지 않아도 자동으로 조인에 사용된다
동일한 이름을 갖는 컬럼이 있지만 데이터 타입이 다르면 에러가 발생한다
조인하는 테이블간의 동일 컬럼이 SELECT 절에 기술되도 테이블 이름을 생략한다
SELECT DEPARTMENT_ID 부서,DEPARTMENT_NAME 부서이름,LOCATION_ID 지역번호,CITY 도시
FROM DEPARTMENTS
NATURAL JOIN LOCATIONS
WHERE CITY='SEATTLE'

USING
USING절은 조인에 사용될 컬럼을 지정한다
NATURAL 절과 USING절은 함께 사용할수없다
조인에 이용되지 않은 동일 이름을 가진 컬럼은 컬럼명 앞에 테이블명을 기술한다
조인 컬럼은 괄호로 묶어서 기술해야한다
SELECT DEPARTMENT_ID 부서번호,DEPARTMENT_NAME 부서,LOCATION_ID 지역번호,CITY 도시
FROM DEPARTMENTS
JOIN LOCATIONS USING (LOCATION_ID);

LEFT OUTER JOIN
FROM TABLE A LEFT OUTER JOIN TABLE B
ON A.COL=B.COL 
이것과 같은 오라클버전 SQL문법은
FROM TALBE A,TABLE B
WHERE A.COL=B.COL(+)

JOIN 순서
FROM A,B,C
A와 B가 JOIN되고 C와 JOIN 된다

17.서브쿼리
SELECT 스칼라 서브쿼리
FROM 인라인뷰 (메인쿼리의 컬럼 사용가능)
WHERE 중첩 서브쿼리
GROUP BY 사용불가
HAVING 중첩 서브쿼리
ORDER BY 스칼라 서브쿼리

IN 서브쿼리 출력값들 OR 조건
ANY/SOME 서브쿼리 출력값들중 가장 작거나 큰 값과 비교
ALL ANY/SOME과 반대되는 개념
EXISTS 서브쿼리내 SELECT 절에는 뭐가 와도 상관없음 ROW가 있으면 TRUE 없으면  FALSE

18.집합연산자
UNION 정렬O 중복제거O 하지만 느림
INTERSECT 정렬O 교집합 하지만 느림
MINUS(EXCEPT) 정렬O 차집합 하지만 느림
UNION ALL 정렬 X 중복제거 X 하지만 빠름

19.DDL
TRUNCATE - DROP & CREATE 테이블 내부 구조는 남아 있으나 데이터가 모두 삭제된다
DROP 테이블 자체가 없어진다 (데이터도 없어진다)
DELETE 데이터만 삭제
ROLLBACK COMMIT 항상 같이 나온다

20.DML
INSERT 데이터를 넣는 명령, INSERT INTO 테이블 (COL1,COL2,COL3 ...) VALUES (VAL1,VAL2,VAL3 ...)
VALUES를 기준으로 좌우의 괄호속 개수가 맞는지 확인해야함

UPDATE 데이터의 특정 행의 값을 변경한다 (DELETE & INSERT)
UPDATE 테이블 SET COL ='값' WHERE COL1='조건'

DELETE 데이터의 특정행을 삭제한다
DELETE FROM 테이블 WHERE COL='조건'

MERGE 특정 데이터를 넣을때 해당 테이블 키값을 기준으로 있으면 UPDATE 없으면 INSERT를 한다 (최근에 기출된적 있음)

위 문제 모두 COMMIT,ROLLBACK,SAVEPOINT와 주로 함께 출제된다

21.제약조건
PK-NOT NULL + UNIQUE
테이블당 하나의 PK를 가질수있다 (하나라는건 컬럼이 아니다,복합키도 가능함)
NOT NULL 해당 컬럼에 NULL값은 올수없다
UNIQUE 해당 컬럼에 중복값이 올수없다

22.DCL
GRANT REVOKE 문법
GRANT 시스템 권햔명 [, 시스템권한명 ... | 롤명 ] TO 유저명 [,유저명...|롤명... |PUBLIC |WITH ADMIN OPTION];
REVOKE {권한명[,권한명...]ALL} ON 객체명 FROM {유저명[,유저명...]|롤명(ROLE)|PUBLIC} [CASCADE CONSTRAINTS];
ROLE은 객체이다

23.VIEW
독립성,편의성,보안성
SQL을 저장하는 개념이다

24.그룹함수
ROLL UP
GROUP BY 에 있는 컬럼들을 오른쪽에서 왼쪽순으로 그룹을 생성한다
A,B로 묶이는 그룹의 값
A로 묶이는 그룹의 소계
전체합계순

CUBE
나올수있는 모든 경우의 수로 그룹을 생성한다
A,B로 묶이는 그룹의 값
A로 묶이는 그룹의 소계
B로 묶이는 그룹의 소계
전체합계
ROLLUP(A,B)!=ROLLUP(B,A),CUBE(A,B)=CUBE(B,A)

GROUPINGSETS
GROUPING
어떤 결과가 나오고 어떤 함수를 사용했는지에 대한 문제 기출

25.TCL
COMMIT,ROLLBACK
-AUTO COMMIT,BEGIN TRANSACTION(COMMIT기능 잠시끄기)END

26.윈도우 함수
ROWS BETWEEN AND 값이 증가함
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW AS '직업별 합계'
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING AS '위아래 합계'
RANGE BETWEEN AND 값이 동일함

1.UNBOUNDED PRECEDING : 최종 출력될 값의 맨 처음 ROW의 값 (PARTITION BY 고려)
2.CURRENT ROW : 현재의 ROW 값
3.UNBOUNDED FOLLOWING : 최종 출력될 값의 맨 마지막 ROW의 값 (PARTITION BY 고려)

RANK 1,1,3,4...
DENSE_RANK 1,1,2,3... 

PARTITION BY, ORDER BY
ROW_NUMBER() OVER (PARTITION BY COL1 ORDER BY COL2)

27.계층형 함수
PRIOR 자식 데이터 = 부모데이터
부모데이터에서 자식데이터로 가면 순방향
SELECT LEVEL, #순방향
            LPAD('',4*(LEVEL-1))||사원 사원,
            관리자,
            CONNECT_BY_ISLEAF ISLEAF
        FROM 사원
        START WITH 관리자 IS NULL
    CONNECT BY PRIOR 사원 = 관리자;

SELECT LEVEL, #역방향
            LPAC('',4*(LEVEL-1))||사원 사원,
            관리자,
            CONNECT_BY_ISLEAF ISLEAF
        FROM 사원
        START WITH 사원 ='D'
    CONNECT BY PRIOR 관리자 = 사원;

28.PL/SQL
EXCEPTION (생략가능)

PROCEDULE (반드시 값이 안나옴)

TRIGGER (커밋 롤백 안된다)
-BEFORE,AFTER 별로 INSERT,UPDATE,DELETE가 있다

FUNCTION (반드시 반환값이 있다)

29.엔터티
관리해야할 대상이 엔터티가 될수있다
인스턴스는 2개이상
업무에서 사용해야함 (프로세스)
관계를 하나이상 가져야한다

유형엔터티
개념엔터티
사건엔터티
(엔터티 유개사)

기본엔터티
중심엔터티
행위엔터티
(엔터티 기중행)

30.속성
기본속성
설계속성
파생속성
(속성 기설파)

31.도메인
데이터유형
크기
제약조건
-CHECK PRIMARY KEY, FOREIGN KEY, NOT NULL, UNIQUE ... 

32.관계
표현부호
타원,해쉬마크 및 삼지창 
-식별관계는 실선
-비식별관계는 점선
의미 : 0,1또는 그 이상의 개체를 허용한다

삼지창 있는 해쉬마크
식별-실선
비식별-점선
의미:1또는 그 이상의 개체를 허용한다

해쉬마크가 있는 타원
식별-실선
비식별-점선
의미 : 0또는 1개체 허용한다

해쉬마크만 있음
식별-실선
비식별-점선
의미 : 정확히 1개체만 허용한다

#깃허브 32번 그림을 잘 보자

33.식별자
유일성 -유일하게 인스턴스 구분
최소성 -최소한의 컬럼으로
불변성 -값이 바뀌지 않아야한다
존재성 -NOT NULL
(식별자 유최불존)
위 4개를 만족하면 후보키가 될수있으며 그중 하나 대표하는것이 기본키

34.식별자&비식별자
식별자 
-강한관계
-PK가 많아짐 (조인시)
-SQL이 복잡해진다

비식별자
-약한관계
-SQL이 느려진다

35.ERD
그리는방법
-좌측상단에서 우측하단으로
-관계명 반드시 표기하지 않아도된다
-UML은 객체지향에서만 쓰인다

36.성능 데이터 모델링
아키텍쳐 모델링 (먼저)
-테이블 파티션 컬럼 등의 정규화 및 반정규화

SQL 튜닝 (그다음)
JOIN  수행 원리

HASH JOIN
-등가 조인만 사용한다
-HASH 함수를 사용하여 SELECT,JOIN 컬럼을 저장한다 (선행 테이블)
-선행 테이블이 작다
-HASH 처리를 위한 별도의 공간이 필요하다

NL JOIN
-랜덤엑세스
-대용량 정렬(SORT)작업
-선행 테이블이 작을수록 유리하다

SORT MERGE
-JOIN 키를 기준으로 정렬한다
-등가/비등가 JOIN이 가능하다

OPTIMIZER
-CBO 제일 경제적인걸 정한다
-RBO 규칙에 의해서 정한다

37.정규화
1차 원자성확보,기본키 설정
2차 기본키가 2개이상의 속성으로 이루어진 경우 부분함수종속성 제거
3차 기본키 제외 칼럼간의 종속성 제거,이행함수종속성 제거
BCNF : 기본키 제외 후보키가 있는경우 후보키가 기본키를 종속시키면 테이블 분해
4차 여러 칼럼들이 하나의 칼럼을 종속시키는 경우 분해하여 다중값종속성 제거
5차 조인에 의해 종속성이 발생되는 경우 분해

38. 이상현상
1.삽입이상 : 새 데이터를 삽입하기 위해서 불필요한 데이터도 함께 삽입해야 하는 이상 문제
2.갱신이상 : 중복 튜플 중 일부만 변경하여 데이터가 불일치하게 되는 이상 문제
3.삭제이상 : 튜플 삭제하면 필요한 데이터까지 함께 삭제되는 이상 문제

39.반정규화
데이터의 무결성을 해칠수도 있다

절차
대량범위처리 빈도수 조사
범위처리 빈도수 조사
통계처리 여부 조사

종류
테이블 병합 1:1/1:M
슈퍼/서브 타입 병합
부분-통계-중복-부분 테이블 분할
이력-중복-파생컬럼 추가
PK를 일반컬럼으로 병합
응용시스템 오작동피하기 위한 임시값 컬럼 추가
중복관계 추가

40. 데이터에 따른 성능
행 이전
1.UPDATE로 인해 행 길이가 증가했을때 저장공간이 부족한 경우 발생
2.원래 정보를 기존블럭에 남겨두고 실제 데이터는 다른 블록에 저장 -> 검색시 원래 블록에서 주소를 먼저 읽고 다른 블럭을 찾아야하므로 성능감소
3.해결책 : PCTFREE 영역을 충분히 할당한다 -> PCTFREE가 너무 큰 경우 데이터 저장 공간 부족으로 공간 효율성 감소

행 연결
1.데이터가 커서 여러 블럭에 나누어 저장하는 현상 -> 2개 이상의 데이터 블럭을 검색해야 하므로 성능 감소
2.INITIAL ROW PIECE (행 조각)와 ROW POINTER로 블록 내에 저장
3.해결책 DB_BLOCK_SIZE를 크게하여 최소화 가능 ->사이즈 변경이 어렵고 무조건 크게 할수없다

LIST PARTITION
-특정 값 기준으로 나눈다
-관리가 쉽다
-데이터가 치우칠수도 있다

RANGE PARTITION
-특정 값의 범위에 따라 나눈다
-관리가 쉽다
-가장 많이쓴다

HASH PARTITION
-관리가 어렵다

41.슈퍼/서브 타입
1. 1:1 타입 (ONE TO ONE TYPE)
2.슈퍼 + 서브 타입 (PLUS TYPE)
3.ALL IN ONE 타입 (SINGLE TYPE)

1)트랜잭션은 항상 일괄로 처리하는데 테이블은 개별로 유지되어 UNION 연산에 의해 성능이 저하될수있음
2)트랜잭션은 항상 서브 타입 개별로 처리하는데 테이블은 하나로 통합되어 있어 불필요하게 많은 양의 데이터 때문에 성능이 저하된다
3)트랜잭션은 항상 슈퍼+서브타입을 공통으로 처리하는데 개별로 유지되어 있거나 하나의 테이블로 집약되어있어 성능이 저하됨

42.분산 데이터 베이스
분할 투명성 : 사용자가 입력한 전역 질의를 여러개의 단편 질의로 변환해주기 때문에 사용자는 전역 스키마가 어떻게 분할되어있는지 알 필요없음
위치 투명성 : 어떤 작업을 수행하기 위해 분산 데이터베이스 상에 존재하는 어떠한 데이터의 물리적인 위치도 알필요 없음
지역사상 투명성 : 지역 DBMS와 물리적DB사이의 매핑 보장,각 지역시스템 이름과 무관한 이름 사용가능
중복 투명성 : 어떤 데이터가 중복되었는지 또는 어디에 중복 데이터를 보관하고있는지 사용자가 알 필요없음
장애 투명성 : 분산되어있는 각 컴퓨터 시스템이나 네트워크에 장애가 발생하더라도 데이터의 무결성이 보장
병행 투명성 : 다수의 TRANSACTION이 동시 수행시 결과의 일관성 유지, 잠금과 타임스탬프의 두가지방법을 주로 사용한다

단점 : 데이터의 무결성을 해칠수있다

43.인덱스
사용못하는경우
-부정형
-LIKE
-형변환 (묵시적)

악영향
-DML 사용시 성능이 저하된다
