def outer(n1):
    def inner(n1):
        return n1+1
    n2=inner(n1)
    print(n1,n2)
    
outer(3)

# 동적 함수 생성에도 사용
def pow_factory(p):
    def pow(n):
        return n**p
    return pow

powOfTwo=pow_factory(2)
powOfThree=pow_factory(3)
print(powOfTwo(5))
print(powOfThree(5))

# 중첩 함수의 스코프
########
def f():
    x=45
    def g():
        x=145
        print(x) #145
    g()
    print(x) #45

x=3
f()
print(x) #3
################
def f():
    x=45
    def g():
        global x
        x=145
        print(x)  #145
    g()
    print(x) #45
x=3
f()
print(x) #145
######################
def f():
    x=45
    def g():
        nonlocal x
        x=145
        print(x) #145
    g()
    print(x) #145
x=3
f()
print(x) #3
