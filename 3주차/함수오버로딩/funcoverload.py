def add(num1, num2):
    res=num1+num2
    return res

def add(num1, num2,num3):
    res=num1+num2+num3
    return res

# def add(num1, num2,num3=None):
#     if num3 is None:
#         res=num1+num2
#     else:
#         res=num1+num2+num3
#     return res

x=add(3,5) # Error - num of args
print(x)
y=add(3,5,7)
print(y)