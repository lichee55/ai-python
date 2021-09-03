# 상속

class A:
    def __init__(self):self.a=3;self.b=5
    def m1(self):return self.a+self.b
    
class B(A):
    def test1(self):
        print(self.a)
    def test2(self):
        print(str(self.m1()))
