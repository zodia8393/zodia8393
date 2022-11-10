/*
스토어드 프로시저 
	MYSQL에서 제공되는 프로그래밍 기능
	쿼리문의 집합으로 어떤 동작을 일괄 처리하기 위한 용도로 사용
    쿼리 모듈화
		필요할 때마다 호출만 하면 훨씬 편리하게 MYSQL 운영가능
        CALL 프로시저 이름() 으로 호출
        
	형식:
	DELIMITER $$  <-구분자를 ;에서 $$로 변경시켜서 ;에서 코드가 끝나지 않게 한다
    CREATE PROCEDURE 스토어드 프로시저 이름 (IN 또는 OUT 파라미터)
    BEGIN
		원하는 과정 
	END $$
    DELIMITER ;
    CALL 스토어드 프로시저 이름();
    
수정과 삭제
	수정: ALTER PROCEDURE
    삭제: DROP PROCEDURE
    
매개 변수 사용
	입력 매개 변수 지정하는 형식
		IN 입력_매개변수_이름 데이터 형식
    
	입력 매개 변수있는 스토어드 프로시저 실행 방법
		CALL 프로시저 이름(전달값);
    
    출력 매개변수 설정
		지정방법
			OUT 출력 매개변수 이름 데이터형식
        출력 매개변수에 값 대입하기 위해 주로 SELECT...INTO 문 사용
        출력 매개변수 있는 스토어드 프로시저 실행방법
			CALL 프로시저_이름(@변수명);
            SELECT @변수명;
		
    
IF...ELSE
	일반적인 SQL에서 변수선언과 다르게 DECLARE로 선언하고 사용
    지역변수는 @를 붙이지 않는다
    
스토어드 프로시저 내의 오류 처리
	DECLARE 액션 HANDLER FOR 오류조건 처리할문장 구문
    EX) DECLARE CONTINUE HANDLER FOR 1146 SELECT "테이블이 없어" AS "메시지";
    

현재 저장된 프로시저 이름 및 내용 확인
	INFORMATION_SCHEMA 데이터 베이스의 ROUTINES 테이블 조회시 확인가능
    SELECT ROUTINE_NAME,ROUTINE_DEFINITION FROM INFORMATION_SCHEMA.ROUTINES
    WHERE ROUTINE_SCHEMA ='SQLDB' AND ROUTINE_TYPE = 'PROCEDURE';
    
스토어드 프로시저 특징
	MYSQL 성능 향상
		긴 쿼리 아니라 짧은 프로시저 내용만 클라이언트에서 서버로 전송
			네트워크 부하 감소로 성능 향상
	
    유저관리가 간편
		응용 프로그램에서는 프로시저만 호출
			데이터베이스에서 관련된 스토어드 프로시저의 내용 수정/유지보수
            
	모듈식 프로그래밍 가능
		언제든지 실행가능
        스토어드 프로시저로 저장해 놓은 쿼리수정,삭제 등 관리 수월
        모듈식 프로그래밍 언어와 동일한 장점 가짐
        
	보안강화
		사용자 별로 테이블 접근 권한 주지 않고 스토어드 프로시저에만 접근 권한 주어 보안강화
			뷰 또한 스토어드 프로시저와 같이 보안 강화 가능
            
	
스토어드 함수
	사용자 지정 함수
    스토어드 프로시저와 유사하나 형태와 사용 용도에 있어 차이 존재
    
    형식
    DELIMITER $$
    CREATE FUNCTION 스토어드 함수명(파라미터)
		RETURNS 반환 형식
	BEGIN
		원하는 코드 입력
		RETURN 반환값;
	END$$
    DELIMITER ;
    SELECT 스토어드함수명();
    
	스토어드 함수와 프로시저의 차이점
    
		스토어드 함수
    
			파라미터에서 IN,OUT등을 사용할수없다
				모두 입력 파라미터로사용한다
			RETURNS 문으로 반환할 값의 데이터 형식을 지정한다
				본문 안에서는 RETURN문으로 하나의 값을 반환한다
			SELECT 문장 안에서 호출을 한다
			함수 안에서 집합 결과 반환하는  SELECT 사용불가
				SELECT...INTO...는 집합결과 반환하는것이 아니므로 스토어드 함수에서 사용가능
			어떤 계산 통해서 하나의 값 반환하는데 주로 사용한다
        
		스토어드 프로시저
			파라미터에서 IN,OUT등 사용가능
			별도의 반환구문이 없음
				필요시 여러개의 OUT파라미터 사용해서 값 반환 가능
			CALL로 호출
			스토어드 프로시저 안에 SELECT문 사용가능
			여러 SQL문이나 숫자 계산 등의 다양한 용도로 사용
        
	스토어드 함수를 사용하기 위해서는 스토어드 함수 생성 권한을 허용해야한다
    SET GLOBAL LOG_BIN_TRUST_FUCTION_CREATORS=1;
    
    현재 저장된 스토어드 함수 이름 및 내용 확인
		SHOW CREATE FUNCTION 함수명;
        
	스토어드 함수 삭제
		다른 데이터베이스 개체와 마찬가지로 DROP문 사용
        DROP FUNCTION 함수명;
	
    
    
	
    
    










    
