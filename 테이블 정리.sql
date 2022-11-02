#데이터베이스 개론 정리
#테이블과 뷰 편

#뷰
#가상의 테이블
#실제 행 데이터를 가지고 있지 않음
#	실체는 없고 진짜 테이블에 링크된 개념
#    뷰를 select => 진짜 테이블의 데이터를 조회하는 것과 동일한 결과

#테이블
#테이블 만들기
#create table 테이블 이름(
#각 컬럼이름 자료형 원하는 제약조건)
/*ex)
drop table if exists buytbl;
create table buytbl(
num int auto_increment not null primary key,
userid char(8) not null,
prodName char(6) not null,
groupName char(4) null,
price int not null,
amount smallint not null);*/

#테이블에 데이터 입력
#insert into 테이블 명 values (데이터);
/*ex
INSERT ignore INTO usertbl VALUES('LSG', '이승기', 1987, '서울', '011', '1111111', 182, '2008-8-8');
INSERT INTO buytbl VALUES(NULL, 'KBS', '운동화', NULL, 30, 2);*/

#제약조건
#데이터의 무결성을 지키기 위한 제한된 조건
#무결성 : 데이터의 정확성과 일관성 유지하고 보증하는 기능
#특정 데이터를 입력시 어떠한 조건을 만족했을때에 입력되도록 제약

# 데이터 무결성을 위한 제약조건

#• PRIMARY KEY 제약 조건
	#기본키
	#	테이블에 존재하는 많은 행의 데이터를 구분할수있는 식별자
	#	중복이나 NULL값 입력 불가
	#	기본키로 생성된것은 자동으로 클러스터형 인덱스 생성
	#	테이블에서는 기본키를 하나 이상 열에 설정 가능
    
    #create table 테이블 이름
    #(각 컬럼이름 자료형 제약조건,
    # constraint primary key 제약조건이름(컬럼이름));
    
#• FOREIGN KEY 제약 조건
	#외래 키
    # 두 테이블 사이의 관계 선언하여 데이터의 무결성 보장해주는 역할
	# 외래 키 관계를 설정하면 하나의 테이블이 다른 테이블에 의존
	# 외래 키 테이블이 참조하는 기준 테이블의 열은 반드시 Primary Key이거나 Unique 제약 조건이 설정되어 있어야 함
	# 외래 키의 옵션 중 ON DELETE CASCADE 또는 ON UPDATE CASCADE
	# 기준 테이블의 데이터가 변경되었을 때 외래 키 테이블도 자동으로 적용되도록 설정

	#create table 테이블 이름
    #(각 컬럼이름 자료형 제약조건,
    # Foreign key(컬럼이르) references 테이블이름(컬럼이름));
    
    #alter table 테이블이름
    #	add constraint 제약조건이름
	#	foreign key (컬럼이름)
    #	references 테이블이름(컬럼이름);
    

#• UNIQUE 제약 조건
	# ‘중복되지 않는 유일한 값’을 입력해야 하는 조건
	# PRIMARY KEY와 비슷하나 UNIQUE는 NULL 값 허용
	# NULL은 여러 개가 입력되어도 상관 없음
	# ex) 회원 테이블 Email 주소 Unique로 설정
	#create table 테이블 이름
    #(컬럼이름 자료형 제약조건(UNIQUE));

#• CHECK 제약 조건(MySQL 8.0.16부터 지원)
	#⁃ 입력되는 데이터를 점검하는 기능
	#alter table문으로 제약조건 추가 가능
    #constraint 제약조건명 check (점검하는 조건)
    
#• DEFAULT 정의
	# 값 입력하지 않았을 때 자동으로 입력되는 기본 값 정의하는 방법
	# ALTER TABLE 사용 시에 열에 DEFAULT를 지정하기 위해서 ALTER COLUMN문 사용
	# create table 테이블명 (컬럼명 자료형 default 원하는 값);
    
#• NULL 값 허용
	# NULL 값을 허용하려면 NULL을, 허용하지 않으려면 NOT NULL을 사용
	# PRIMARY KEY가 설정된 열에는 생략하면 자동으로 NOT NULL
	# NULL 값은 ‘아무 것도 없다’라는 의미, 공백(‘ ‘) 이나 0과 다름

#테이블 압축
#대용량 테이블의 공간 절약하는 효과
/*ex)
create database if not exists 데이터베이스명;
use 데이터베이스명;
create table 테이블명(컬럼명 자료형) ROW_FORMAT=COMPRESSED ;*/
#데이터베이스를 압축하기 때문에 일반 데이터베이스보다 시간이 더 오래걸린다

#임시테이블
#create temporary table [if not exists] 테이블명 (열 정의);
#임시로 잠깐 사용되는 테이블
#세션내에서만 존재(세션이 닫히면 자동삭제)
#생성한 클라이언트에서만 접근 가능(다른 클라이언트에서는 접근 불가)
#임시테이블 삭제시점(drop table로 삭제,워크벤치종료,클라이언트종료,MySQL재시작)

#테이블 삭제
#DROP TABLE 테이블명;
#외래키 제약조건의 기준 테이블은 삭제불가
#동시삭제 가능

#테이블 수정
#alter table 테이블명 추가하고자 하는 조건;
#열 추가
#alter table add 컬럼명 자료형;
	#기본적으로 가장 뒤에 추가
	#순서 지정하려면 제일뒤에 FIRST 또는 ALTER 열 이름 지정

#열 삭제
#alter table 테이블명 drop column 컬럼명;
	#제약조건이 걸린 열 삭제할경우 제약조건 먼저 삭제 후 열을 삭제해야함

#열 이름 및 데이터 형식 변경
#alter table 테이블명 change column 기존컬럼명 원하는컬럼명 원하는자료형 원하는 제약조건;

#열 제약조건 추가 및 삭제
#alter table 테이블명 drop 제약조건;
	#기본키 또는 외래키로 연결되어있으면 해당 조건을 먼저 제거후 제거해야한다

#테이블 데이터 업데이트
#update 테이블명 set 원하는 조건;
#foreign key 업데이트시 참조 데이터로 인해 업데이트가 안됨
	#시스템변수로 강제 변환 set foreign_key_checks=0;
	#on cascade 사용
		#참조테이블의 변화를 자동반영한다
        


