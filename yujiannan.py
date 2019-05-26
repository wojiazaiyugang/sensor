from enum import IntEnum
class StabilityLevel(IntEnum):
    NOT_WALKING = 1,
    WALK_BUT_NO_CYCLE = 2,
    CYCLE_DETECTED = 3,

for i in StabilityLevel:
    print(i)