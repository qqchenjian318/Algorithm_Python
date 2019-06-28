# coding=utf-8

import time


# 剑指offer的算法题 python实现


class ListNode(object):
    def __init__(self, value=None, nextNode=None):
        self.value = value
        self.nextNode = nextNode


# 1、二维数组的查找
# 在一个二维数组中，每一个行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
# 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
# 例如下面的二维数组就是每行、每列都递增排序。如果在这个数组中查找数字7，则返回true；
# 如果查找数字5，由于数组不含有该数字，则返回false
# 默认数一个规则的二维数组（矩形）
# 1 2 8 9
# 2 4 9 12
# 4 7 10 13
# 6 8 11 15
# 思路：
# 从一个角开始 一行一行的查找 一片片的排除不符合要求的区域
# 如果从左上角或者右下角开始（是非常复杂的 并不能很好的完成需求
# 无法对行 或者列进行排除）
# 所以 我们从右上角开始 ，如果当前小于value 那么可以排除当前行
# 如果当前大于value 那么可以排除当前列
# x_index和y_index就可以进行随意的移动
# 所以这道题非常容易陷入 从左上角或者右下角进行查询的错误路线
# （！！！我靠 第一题就陷入了陷阱了 残忍）

def find_num(L=[], value=0):
    # 从数组L中查找 value
    # 从
    if L is None:
        return False
    # 从第一行开始查找
    max_x = len(L)
    max_y = len(L[0])
    x = 0
    y = max_y - 1
    while x < max_x and y >= 0:
        if L[x][y] == value:
            return True
        elif L[x][y] > value:
            # 说明 可以排除 y所在的行
            y -= 1
        else:
            # 说明当前数比value要小 那么可以排除当前行
            x += 1
    return False


# check code
# arr = [[1, 2, 8, 9], [2, 4, 9, 12], [4, 7, 10, 13], [6, 8, 11, 15]]
# print('查找 7 result=%s ' % find_num(arr, 7))
# print('查找 5 reslut=%s ' % find_num(arr, 5))
# print('查找 15 reslut=%s ' % find_num(arr, 15))
# print('查找 11 reslut=%s ' % find_num(arr, 11))
# print('查找 3 reslut=%s ' % find_num(arr, 3))

# -------------------------------------------------------

# 2、输入一个链表的头结点，从尾到头反过来打印每个结点的值。
#
# 单项链表的定义
# class Node{
#    int value
#    Node nextNode
# }
# 思路一：用一个数组存放所有的链表值 然后倒序打印
# 思路二：用递归实现(如果链表过长 会出现嵌套太深
# 可能导致栈溢出的情况出现)
# 思路三：改变链表的指向 然后打印

def print_list(firstNode=None):
    if firstNode is None:
        print(' 链表是None哟')
        return
    values = []

    while True:
        values.append(firstNode.value)

        firstNode = firstNode.nextNode
        if firstNode is None:
            break

    while len(values) > 0:
        print(values.pop())


def print_list_2(firstNode=None):
    if firstNode is None:
        print('链表是None哟')
        return
    print_value(firstNode)


def print_value(node=None):
    if node.nextNode is not None:
        print_value(node.nextNode)

    print(node.value)


# check code
# five = ListNode(5, None)
# four = ListNode(4, five)
# three = ListNode(3, four)
# two = ListNode(2, three)
# one = ListNode(1, two)
#
# print_list_2(one)

# -------------------------------------------------------

# 3、用两个栈（先进后出）实现一个队列（先进先出）或者用两个队列实现一个栈
# 函数 isEmpty pop push length
# 先利用数组实现一个栈 然后用两个栈 实现一个队列
#
class Stack(object):
    def __init__(self):
        self.stack = []

    def push(self, value=None):
        if value is not None:
            self.stack.append(value)

    def pop(self):
        if len(self.stack) > 0:
            return self.stack.pop()

    def isEmpty(self):
        return len(self.stack) == 0

    def length(self):
        return len(self.stack)

    def values(self):
        return self.stack


class Queue(object):
    def __init__(self):
        self.inputStack = Stack()
        self.outputStack = Stack()

    def push(self, value=None):
        if value is not None:
            # 如果输出栈 不为空 则先把所有的数据
            # 放入到输入栈中 再存入本次数据
            if not self.outputStack.isEmpty():
                while self.outputStack.length() > 0:
                    self.inputStack.push(self.outputStack.pop())

            self.inputStack.push(value)

    def pop(self):
        # 取数据的时候 先将所有inputStack的数据 放入到outputStack 再取 取完之后
        # 再将outputStack的数据 放入到inputStack
        print('取出前的状态 input=%s output=%s ' % (self.inputStack.values(), self.outputStack.values()))
        if not self.inputStack.isEmpty():
            # 如果取的不为空 则先全部添加到另一个stack 再取
            while self.inputStack.length() > 0:
                self.outputStack.push(self.inputStack.pop())
            return self.outputStack.pop()
        else:
            return self.outputStack.pop()

    def isEmpty(self):
        return self.outputStack.isEmpty() and self.inputStack.isEmpty()

    def length(self):
        outL = self.outputStack.length()
        inL = self.inputStack.length()
        if outL > inL:
            return outL
        else:
            return inL


# check code
# stack = Stack()
# stack.push(1)
# stack.push(2)
# stack.push(3)
# print(stack.pop())
# stack.push(4)
# print(stack.pop())
# print(stack.pop())
# print('当前栈的长度 %s' % stack.length())
# queue = Queue()
# queue.push(1)
# queue.push(2)
# queue.push(3)
# print(queue.pop())
# queue.push(4)
# print(queue.pop())
# print(queue.pop())
# print('当前队列的长度 %s' % queue.length())

# -------------------------------------------------------
# 4、把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
# 输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。
# 例如数组{3,4,5,1,2} 为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
# 数组里面可能存在重复的值
# 提供校验的数组[3,4,5,1,2] [1,0,1,1,1],[1,2,0,1,1]
# 思路：
# 相对于对简单的有序数组进行排序来说，这个题 更加复杂一点
# 首先 输入的数组是有一定规律的
# 我们需要时间复杂度最低的方式 找到该数组的最小值
# 还是采用二分法
# start   mid    end
# if start < mid and mid < end
#   不可能
# elif start < mid and mid > end
#   说明小值再后半部分
# elif start > mid and mid < end
#   说明 小值在前半部分
# elif start > mid and mid > end
#   不可能
# elif start == mid and mid < end
#   不可能
# elif start == mid and mid > end
#   说明小值 在后半部分
# elif start > mid and mid = end
#   说明 小值 在前半部分
# elif start < mid and mid = end
#   不可能
# elif start == mid and mid = end
#   无法判定 极值在前半部分还是在后半部分 所以采用顺序的方式


def find_min(L=[]):
    if len(L) == 0:
        print('输入的数组是None的哟')
        return

    start = 0
    end = len(L) - 1
    mid = (start + end) // 2
    while start < mid < end:

        print('开始比较了哟 %s %s %s' % (L[start], L[mid], L[end]))
        if L[start] <= L[mid] and L[mid] > L[end]:
            # 说明值在后半部分
            start = mid
            mid = (start + end) // 2
        elif L[start] > L[mid] and L[mid] <= L[end]:
            # 说明值在前半部分
            end = mid
            mid = (start + end) // 2
        elif L[start] == L[mid] and L[mid] == L[end]:
            # 当三个值相同的时候 无法判定 极值在前还是在后 只有采用顺序查找的方式

            return find_min_in_order(L)
        if end - start == 1:
            mid = end
            return L[mid]

    return L[mid]


def find_min_in_order(L=[]):
    min = L[0]
    for i in L:
        if i < min:
            min = i
    return min


# check code
# arr = [1, 2, 0, 1, 1]
# print('查找数组的极值 %s ' % (find_min(arr)))


# -------------------------------------------------------
#
# 5、斐波那契数列，写一个函数，输入n，求斐波那契数列（Fibonacci）的第n项。斐波那契数列的定义如下
#       n = 0  = 0
# f(n)  n = 1  = 1
#       n = n  = f(n - 1) + f( n - 2)
# 0 1 1 2 3 5 8 13 21
# 思路一：
#   递归，存在严重的效率问题，比如计算far(8)的时候
#                       f(5)
#            f(4)                    f(3)
#       f(3)     f(2)           f(2)       f(1)
#    f(2) f(1)  f(1) f(0)    f(1)  f(0)
# f(1) f(0)
#    从上面的逻辑看 会存在很严重的重复计算
#
# 思路二：
#   避免掉上述的重复计算 计算过的数 使用一个字典记录下来
#
# 思路三：
#   为了避免重复计算，我们可以使用顺序计算的方式，从f(0)开始计算，并且每次
#   保存前两个数的值，这样一次遍历 就可以完成计算了
#

# 这种方式 如果计算比较大大值 比如超过30 就会出现严重的效率问题 耗时很久 而且倍数增长的
def far(value=0):
    if value == 0:
        return 0
    elif value == 1:
        return 1

    return far(value - 1) + far(value - 2)


# 这种方式 缓存了已经计算过的 就在计算大值的时候 比如100 也不会有明显的耗时
def far_1(value=0, arr={}):
    print('新计算  %s ' % value)
    if value == 0:
        arr[0] = 0
        return 0
    elif value == 1:
        arr[1] = 1
        return 1
    if arr.get(value) is not None:
        # 说明已经计算过了 不再进行计算
        print('计算过了 不再计算 %s ' % (value - 1))
        return arr.get(value)

    if arr.get(value - 1) is not None:
        first = arr.get(value - 1)
        print('计算过了 不再计算 %s ' % (value - 1))
    else:
        first = far_1(value - 1, arr)

    if arr.get(value - 2) is not None:
        second = arr.get(value - 2)
        print('计算过了 不再计算 %s ' % (value - 1))
    else:
        second = far_1(value - 2, arr)

    arr[value] = (first + second)
    return first + second


# 和上种算法的思想类似 都是避免重复计算
# 但是比上面的算法 更加节约内存空间 而且效率更高
# 时间复杂度为O(n)
# 逻辑就是 从f(0) 开始计算 f(1) f(2) f(3)...
# 并且每次只保存 当前数的前两个数的值，这样 就可以通过一次遍历 完成整个逻辑的计算

def far_2(value=0):
    startArr = [0, 1]
    if value < 2:
        return startArr[value]
    i = 2
    cur = 1
    lastValue = 1
    lastLastValue = 0
    while i <= value:
        cur = lastValue + lastLastValue

        lastLastValue = lastValue
        lastValue = cur
        i += 1
    return cur


# start = time.time()
#
# print('斐波那契数 %s  %s  %s  %s ' % (far_2(0), far_2(1), far_2(4), far_2(36)))
# end = time.time()
# print('耗时  %s ' % (end - start))


# -------------------------------------------------------
#
# 6、二进制中 1 的个数
# 请实现一个函数，输入一个整数，输出该数二进制表示中1的个数。
# 例如把9表示成二进制是1001，有2位是1.因此如果输入9，该函数输出2.
#
# 思路：
# 最笨的方式 就是遍历一边二进制的char
# 我们当然是高级程序员 所以要使用位运算来计算咯
# 所以 基本思路是 先看最后一位是不是 1
# 然后将数 右移一位 继续看最后一位 即可
# 如果这是一个负数呢？python中没有无符号数
# 所以 如果输入的是负值 需要先处理一下
#

def find_1(value):
    count = 0
    if value < 0:
        value = value & 0xffffffff
    while value:
        count += value & 1
        value >>= 1

    return count


# print('数的位移 %s %s %s %s' % (find_1(9), find_1(-9), find_1(8), find_1(-8)))

# -------------------------------------------------------
#
# 7、数值的n次方
# 实现函数 power(double base，int n)，求base的n次方。不得使用函数库，
# 同时不需要考虑大数问题。
#
# 思路1：
#   不需要考虑大数问题，意思就是不需要考虑超过限制的数
#   n次方原则上就是 base * base * base ...
#   需要考虑 n为负数 正数 0 的情况
#   正数 ，就直接算
#   负数，先算指数的绝对值 再算 次方 最后求倒数
#   0 如果n = 0 那么结果恒等于1
#   还需要考虑 base为0的情况
# 思路二：
#   那这题还有没有优化的空间呢？如果我们要求 2 的4次方
#   那是不是 如果我们算出了2的2次方=4 然后再算一次4的二次方即可
#   这样的话 我们就可以只用循环两次即可 大大的减少了时间
#   也就是说 其实我们可以用递归来求

def power_1(base, n):
    i = 0
    flag = 0
    if base == 0:
        return 0
    if n == 0:
        return 1
    elif n < 0:
        n = -n
        flag = 1
    result = 1
    while i < n:
        result *= base
        i += 1
    if flag:
        result = 1 / result
    return result


# 思路二 减少循环次数

def power_2(base, n):
    if base == 0:
        return 0
    flag = 0
    if n < 0:
        n = -n
        flag = 1
    result = power(base, n)
    if flag:
        result = 1 / result
    return result


def power(base, n):
    print('递归 %s  %s ' % (base, n))
    if n == 0:
        return 1
    if n == 1:
        return base
    nextN = n >> 1
    result = power(base, nextN)
    result *= result
    if n & 0x1 == 1:
        # 说明n是奇数 需要乘一次原始值
        result *= base
    return result


# 思路三：
#   虽然上面使用递归减少了循环次数，但是递归是一个并不怎么高效的方式
#   所以 能不能使用普通循环来完成呢
#
def power_engine(base, n):
    if n == 0:
        return 1
    if n == 1:
        return base
    i = n
    result = base
    while i > 1:
        print('单次循环 %s %s ' % (base, i))
        result *= result
        i >>= 1
    if n & 0x1 == 1:
        result *= base
    return result


def power_3(base, n):
    if base == 0:
        return 0
    flag = 0
    if n < 0:
        n = -n
        flag = 1
    a = power_engine(base, n)
    if flag:
        a = 1 / a
    return a


# print(power_3(2, 5))
# print("数的n次方 %s %s %s %s %s " % (power_2(2, 1), power_2(2, 3), power_2(2, -1), power_2(2, 0), power_2(4, 2)))


# -------------------------------------------------------
# 8、打印1到最大的n位数
# 输入数字n，按顺序打印出从1最大的n位十进制数。比如输入3，则打印出1,2,3... 一直到最大的3位数即999。
# 不考虑n为负数的情况 n 大于等于1
# 先算出该位的最大值 然后遍历
# 但是如果是一个非常大的数 (是不是会出现数字精度不够)
# 所以需要使用字符串 模拟数字加法
# 需要考虑 前面全是0的情况
def print_num(n):
    if n <= 0:
        print('输入错误的位数哟')
        return
    i = 0
    result_num = {}
    while i < n:
        # 当小于的时候 就往数据中添加一位9
        result_num[i] = 0
        i += 1
    print(result_num)
    length = 1
    while not increment(result_num):
        print_arr(result_num)


# 实现在dict上+1
# 如果已经是最大数了 就return true
def increment(L={}):
    isOverflow = False
    take_over = 0
    length = len(L) - 1

    while length >= 0:
        L[length]

        length -= 1




def print_arr(L={}, n=0):
    result = ''
    i = 0
    print('打印的数组 %s  长度：%s  %s ' % (L, i, n))
    while i < n:
        result = result + str(L[i])
        i += 1

    print(result)


print_num(2)
