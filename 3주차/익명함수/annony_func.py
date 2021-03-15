def add1(v1,v2):return v1+v2
sum1=add1(4,5)

add2=lambda v1,v2:v1+v2
sum2=add2(4,5)

print(sum1,sum2)

def makeIncresementor(n) : return lambda x:x+n
f=makeIncresementor(1)
newval=f(5)
print(newval)

g=makeIncresementor(2)
newval=g(5)
print(newval)

# 익명함수의 사용

def f(x):return (x%2)==0
print(list(filter(f,range(-5,5))))

# 위의 방법으로 해도 ㄱㅊ지만
# 필터조건은 익명으로 자주 사용
print(list(filter(lambda x:(x%2)==0,range(-5,5))))

example=[1,2,3,4,5,6]
print(list(map(lambda x:x*x,example)))
