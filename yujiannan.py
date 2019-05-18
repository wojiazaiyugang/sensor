class A:
    a = [1,2,3]

def f(a):
    a = a + [1]

x  =A()
f(x.a)
print(x.a)