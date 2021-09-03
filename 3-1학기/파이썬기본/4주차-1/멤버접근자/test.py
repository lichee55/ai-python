class FirstClass:
    def __init__(self):
        self.__field1=-1
        self.field1=0
    def method1(self):
        self.field1=1
        self.field1=3
    def method2(self):
        print(self.__field1)
        print(self.field1)
        
ob=FirstClass()
print(ob.field1)
# print(ob.__field1)
print(ob.__dict__)
print(ob._FirstClass__field1)