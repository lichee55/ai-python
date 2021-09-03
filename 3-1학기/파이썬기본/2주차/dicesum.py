# 20210308 Mon
# 인공지능 파이썬 수업 2주차

import random

x=random.randint(1,6)
y=random.randint(1,6)

# print(x, y)

a=input('a 예측 : ')
b=input('b 예측 : ')

a=int(a)
b=int(b)

print('첫 주사위 : ',x,end=" ")
print('두번째 주사위 : ',y,end=" ")
print('합은 ',x+y)

if(((x+y)-a)**2>((x+y)-b)**2):
    print("b승")
elif(((x+y)-a)**2<((x+y)-b)**2):
    print("a승")
else:
    print("무승부")
