# 부모클래스의 생성자/ 소멸자는 작동하지않는것을
# 볼 수 있음

class Shape:
    def __init__(self):
        print("Shape 생성자")
        self.x = 0
        self.y = 0
    def __del__(self):
        print("Shape 소멸자")

# class Rectangle(Shape):
#     def __init__(self):
#         print("Rectangle 생성자")
#         self.width = 0
#         self.height = 0
#     def __del__(self):
#         print("Rectangle 소멸자")

# r1=Rectangle()
# ==========================================
# 이렇게하면 됨

class Rectangle(Shape):
    def __init__(self):
        super().__init__()
        print("Rectangle 생성자")
        self.width = 0
        self.height = 0
    def __del__(self):
        print("Rectangle 소멸자")
        super().__del__()

r2=Rectangle()