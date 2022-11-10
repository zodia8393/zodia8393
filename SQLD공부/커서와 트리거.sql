/*커서
개요
	스토어드 프로시저 내부에서 사용
	일반 프로그래밍 언어의 파일처리와 방법이 비슷하다
		행의 집합을 다루기 편리한 기능 제공
	테이블에서 여러개의 행을 쿼리한후,쿼리의 결과인 행 집합을 한 행씩 처리하기 위한 방식

처리순서
	1.커서의 선언(DECLARE CURSOR)
	2.반복 조건 선언 (DECLARE CONTINUE HANDLER)
	3.커서 열기(OPEN)
	4.커서에서 데이터 가져오기(FETCH)
	5.데이터처리
	6.커서 닫기(CLOSE)
    
    4,5번 과정은 LOOP~END LOOP문으로 반복구간을 지정한다

ex)
DROP PROCEDURE IF EXISTS GRADEPROC;
DELIMITER $$
CREATE PROCEDURE GRADEPROC()
BEGIN
	DECLARE ID VARCHAR(10);
    DECLARE HAP BIGINT;
    DECLARE USERGRADE CHAR(5);
    
    DECLARE ENDOFROW BOOLEAN DEFAULT FALSE;
    
    DECLARE USERCURSOR CURSOR FOR  ---커서 선언
		SELECT U.USERID,SUM(PRICE*AMOUNT)
			FROM BUYTBL B
				RIGHT OUTER JOIN USERTBL U
                ON B.USERID=U.USERID
			GROUP BY U.USERID,U.NAME;
	
    DECLARE CONTINUE HANDLER
		FOR NOT FOUND SET ENDOFROW=TRUE;
        
	OPEN USERCURSOR; --커서 열기
    GRADE_LOOP:LOOP
		FETCH USERCURSOR INTO ID,HAP;
        IF ENDOFROW THEN
			LEAVE GRADE_LOOP;
		END IF;
	CASE 
		WHEN (HAP>=1500) THEN SET USERGRADE 'A';
		WHEN (HAP>=1000) THEN SET USERGRADE ='B';
        WHEN (HAP>=1) THEN SET USERGRADE 'C';
        ELSE SET USERGRADE='D';
	END CASE;
    
    
    UPDATE USERTBL SET GRADE = USERGRADE WHERE USERID=ID;
	END LOOP GRADE_LOOP;
    
    CLOSE USERCURSOR;
END $$
DELIMITER ;
    
    
트리거

개요
	사전적 의미로 방아쇠
	테이블에 무슨일이 일어나면 자동으로 실행
    제약 조건과 더불어 데이터 무결성 위해 MYSQL에서 사용가능한 기능
    테이블에 DML문 이벤트가 발생될때 작동
    테이블에 부착되는 프로그램 코드
    직접 실행은 불가능하다
		테이블에 이벤트가 일어나야 자동실행된다
	IN,OUT 매개변수를 사용할수없다
    MYSQL은 VIEW에 트리거를 부착할수없다

EX)
	DROP TRIGGER IF EXISTS TESTTRG;
    DELIMITER //
    CREATE TRIGGER TESTTRG -트리거명
		AFTER DELETE -삭제후 작동하도록 지정
        ON TESTTBL -트리거 부착할 테이블
        FOR EACH ROW -각 행마다 적용
	BEGIN
		SET @MSG ='가수그룹삭제' ; -트리거 실행시 작동되는 코드들
	END//
    DELIMITER ;
    
    
트리거 종류
	AFTER 트리거
		테이블에 INSERT UPDATE DELETE 등의 작업이 일어났을때 작동
		해당 작업후에 작동
	BEFORE 트리거
		이벤트가 발생하기 전에 작동
		INSERT,UPDATE,DELETE 세가지 이벤트로 작동
        
트리거 문법
	CREATE
		DEFINER=USER
        TRIGGER TRIGGER_NAME
        TRIGGER_TIME TRIGGER_EVENT
        ON TBL_NAME FOR EACH ROW
        TRIGGER_ORDER
        TRIGGER_BODY
        
	TRIGGER_TIME : {BEFORE|AFTER}
    TRIGGER_EVENT : {INSERT|UPDATE|DELETE}
    TRIGGER_ORDER : {FOLLOWS|PRECEDES} OTHER_TRIGGER_NAME
    

트리거의 사용
	AFTER 트리거 사용
		TRUNCATE TABLE 테이블이름
			DELETE FROM 테이블명 과 동일효과
            DML문이 아니라 트리거 작동시키지 않음
            
		SIGNAL SQLSTATE '45000'문
			사용자가 오류를 강제발생시키는 함수
			사용자가 정의한 오류메시지 출력
			사용자가 시도한 INSERT는 롤백된다
            
		트리거가 생성하는 임시테이블
			INSERT,UPDATE,DELETE작업 수행되면 임시사용하는 시스템 테이블
            이름은 'NEW'와'OLD'
		작동 개념
		INSERT(새값) -> NEW테이블의 새값 임시생성 -> 기존테이블의 새값
        DELETE(예전값) -> 기존테이블의 예전값 삭제 -> OLD테이블의 예전값 생성
		UPDATE(새값,예전값) ->NEW테이블의 새값 임시생성->기존테이블의 예전값 삭제후 새값추가 ->OLD테이블에 예전값 추가
        


	BEFORE 트리거 사용
		테이블에 변경이 가해지기 전 작동
        활용 예
			BEFORE INSERT 트리거 부착시 입력될 데이터 값을 미리 확인하여 문제 있을경우 다른값으로 변경
            
		SHOW TRIGGERS 문으로 데이터베이스에 생성된 트리거 확인가능
			SHOW TRIGGERS FROM 데이터베이스명;
            
		트리거 삭제
			DROP TRIGGER 트리거명;
            
		
	기타 트리거 관련 내용
		다중 트리거
			하나의 테이블에 동일한 트리거가 여러개 부착되어있는것
            
		중첩 트리거
			트리거가 또 다른 트리거를 작동시키는것
            
		트리거 작동순서
			하나의 테이블에 여러개의 트리거 부착된경우 트리거 작동순서 지정가능
            {FOLLOWS|PRECEDES} 다른 트리거 이름















        

