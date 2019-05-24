from enum import Enum, unique
@unique
class A(Enum):
    aaa = 1
    aaa1= 1

print(type(A.aaa))