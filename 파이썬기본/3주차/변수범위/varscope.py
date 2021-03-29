total = 0

# 함수 정의
def add( arg1, arg2 ):
    total = arg1 + arg2; 
    print("함수 내 : ", total)
    return total;
# 함수 호출
add(10,20);
print("함수 밖 : ", total)

def add( arg1, arg2 ):
    global total  # global 키워드를 사용하여 전역 변수임을 알려줘야 함
    total = arg1 + arg2; 
    print("Inside the function local total : ", total)
    return total;

# 함수 호출
add(10,20);
print("Outside the function global total : ", total)
