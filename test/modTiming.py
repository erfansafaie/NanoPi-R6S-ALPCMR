
import math

with open("lpMain/timingData.csv", "r") as f:
    data = f.readlines()
    del data[0]

for d in data:
    d = d.removesuffix("\n")
    a = d.split(",")
    num = float(a[0] + a[1] + a[2] + a[3] + a[4])
    print(num)

# with open()