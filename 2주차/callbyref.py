i=3;

def change(i):
    i+=1
    return i

print(i)
change(i)
print(i)

def swap(i,j):
    temp=i
    i=j
    j=temp
    return i,j

a=1;b=2

print(a,b)
a,b=swap(a,b)
print(a,b)