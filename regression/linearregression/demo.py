def getmin(arr):
    res = arr[0]

    for i in range(1, len(arr)):
        if res > arr[i]:
            res = arr[i]

    return res


def getmax(arr):
    res = arr[0]

    for i in range(1, len(arr)):
        if res < arr[i]:
            res = arr[i]

    return res


def getaverage(arr):
    res = 0

    for x in arr:
        res += x

    return res / len(arr)


def main():
    arr = []
    print("Nhap 10 so")

    for i in range(10):
        x = float(input(f"Nhap arr[{i}]= "))
        arr.append(x)

    print('Array= ', arr)
    print('Min=', getmin(arr))
    print('Max=', getmax(arr))
    print('Average=', getaverage(arr))

if __name__ == "__main__":
  main()