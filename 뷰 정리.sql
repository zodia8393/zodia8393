#뷰
개념
일반 사용자 입장에서 테이블과 동일하게 사용하는 개체
	뷰 생성후 테이블처럼 접근 가능하여 동일한 결과 얻을수 있음

뷰의 생성

생성 구문
use 데이터베이스 명
create view 뷰이름
as select 원하는 컬럼명 from 테이블명

뷰의 장점
	보안에 도움
		사용자가 중요한 정보에 바로 접근하지 못한다
	복잡한 쿼리 단순화
		긴 쿼리를 뷰로 작성, 뷰를 테이블처럼 사용 가능
        
원본 뷰 덮어쓰기도 가능하다
ALTER VIEW 뷰이름
AS SELECT 원하는 컬럼명(또는 조건) FROM 테이블명

뷰 업데이트
	기존 뷰를 대치한다
		USE 데이터베이스 명;
		CREATE OR REPLACE VIEW 뷰 이름
        AS 원하는 컬럼명(또는 조건) FROM 테이블명 ;
        
	뷰 내용물 확인하기
		DESCRIBE 뷰이름
    
    뷰의 값을 업데이트해도 원본 테이블의 값은 업데이트 되지 않는다
	뷰의 값을 업데이트할때 원본 테이블의 조건을 갖추지 못하면 업데이트가 되지 않는다
    뷰에 있는 필드만 입력하게되면 원본 테이블의 비어있는 컬럼이 발생하는데 NULL값을 허용하지 않으면 에러가 발생한다
    
    그룹함수를 포함하는 뷰
		당연히 집계된 뷰는 값 업데이트가 안된다
        INFORMATION_SCHEMA.VIEWS에서 시스템 데이터베이스에 있는 모든것이 나와있음
        이외에도 UNION,JOIN등을 사용한뷰, DISTINCT,GROUP BY등을 사용한 뷰는 업데이트 불가
        
	조건을 가지는 뷰
		그룹함수를 포함하는 뷰와 같음
        WITH CHECK OPTION을 통해서 뷰의 조건에 맞는 데이터만 추가하도록 설정가능
			-> 조건에 맞지 않을시 에러 발생
            
테이블스페이스
	물리적인 공간을 의미
    데이터베이스는 논리적 공간이다
    테이블 스페이스를 지정하지 않은 경우
		시스템 테이블스페이스에 테이블이 저장된다
	시스템변수 INNODB_DATA_FILE_PATH에 관련내용이 저장된다
    
    테이블 스페이스 파일은 MYSQL SERVER$DATA 폴더에 저장되어있다
    
	성능 향상을 위한 테이블스페이스 추가
		소용량의 데이터를 사용하는 경우에는 테이블스페이스 고려하지않아도 되나
		대용량 데이터 운영시 성능 향상 위해 테이블스페이스의 분리 적극 고려
	
		각 테이블 이 별도의 테이블스페이스에 저장되도록 시스템 변수 INNODB_FILE_PER_TABLE이 ON으로 설정되어야함
        확인 방법 -> SHOW VARIABLES LIKE 'INNODB_FILE_PER_TABLE'
        
		CREATE TABLESPACE 테이블스페이스명 ADD DATAFILE '테이블스페이스 파일명'
        
        각 테이블스페이스에서 파일 생성하기
			USE DB명;
            CREATE TABLE 테이블명 (원하는 컬럼명 자료형) TABLESPACE 테이블스페이스명;
            
		테이블 만든후 ALTER TABLE문으로 테이블스페이스 변경가능
			CREATE TABLE 테이블명 (컬럼명 자료형)
            ALTER TABLE 테이블명 TABLESPACE 테이블스페이스명
            
		대용량 테이블 복사해서 테이블스페이스를 지정
        DROP TABLE 테이블명;
        CREATE TABLE 테이블명 (SELECT * FROM 대용량테이블명);
        ALTER TABLE 테이블명 TABLESPACE 테이블스페이스명;
        
        
    
    
    
    
    
    
    
    
    
    
        
        
        