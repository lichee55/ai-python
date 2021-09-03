# 메소드 재정의

class A :
	def __init__(self) : self.a = 3; self.b = 5
	def m1(self) : return self.a + self.b
class B(A) :
	def m1(self, n) :
		k = super().m1()
		return k + n
	def test1(self) :
		print(self.a)
	def test2(self) :
		print(str(super().m1()))

ob=B()
ob.test1()
ob.test2()