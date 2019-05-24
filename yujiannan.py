a = 11

class A:
    def __init__(self):
        self.bbb = a
        print(id(self.bbb))

b =A()
a = 33
print(id(a))