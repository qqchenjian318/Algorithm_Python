# coding=utf-8

import time


# 剑指offer的算法题 python实现


class ListNode(object):
    def __init__(self, value=None, nextNode=None):
        self.value = value
        self.nextNode = nextNode


# 树的节点
class TreeNode(object):
    def __init__(self, value=None, leftNode=None, rightNode=None):
        self.value = value
        self.leftNode = leftNode
        self.rightNode = rightNode


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
        pass


def print_arr(L={}, n=0):
    result = ''
    i = 0
    print('打印的数组 %s  长度：%s  %s ' % (L, i, n))
    while i < n:
        result = result + str(L[i])
        i += 1

    print(result)


# print_num(2)
# todo 没写完


# -------------------------------------------------------
#
# 9、在O(1)时间删除链表结点
# 给定单向链表的头指针和一个结点指针，定义一个函数在O(1)时间删除该结点。
# 单向链表
# A-》B—》C—》D—》E
# 比如给了A 和C 现在要在O(1)的时间复杂度中删除掉C
# 那就改变C的值 并且将C的下个节点 指向E节点，
# 1、C节点刚好是最后一个节点
# 2、链表只存在C一个节点

def delete_node(start_node, delete_node):
    if start_node == delete_node:
        # 说明只存在一个节点
        start_node = None
    if delete_node.nextNode is not None:
        # 说明不是最后一个节点
        delete_node.value = delete_node.nextNode.value
        delete_node.nextNode = delete_node.nextNode.nextNode
    else:
        # 说明要删除的是最后一个节点 那么就顺序遍历 进行删除
        next_node = start_node
        while next_node is not None and next_node.nextNode is not None:
            if next_node.nextNode == delete_node:
                # 说明到了最后一个节点了
                next_node.nextNode = delete_node.nextNode
            next_node = next_node.nextNode

    return start_node


# E = ListNode(5, None)
# D = ListNode(4, E)
# C = ListNode(3, D)
# B = ListNode(2, C)
# A = ListNode(1, B)
#
# start = delete_node(A, E)
# while start is not None:
#     print(start.value)
#     start = start.nextNode

# -------------------------------------------------------
#
# 10、调整数组顺序使得奇数位于偶数前面
# 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
# 使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分
# 例如 [1,3,4,2,6,7,9] 调整顺序后为 [1,3,7,9,4,2,6]
# 思路：
#   两个指针 start end  如果start为偶数 end为奇数 就交互
#   如果start为奇数 就往后遍历 直到找到第一个奇偶数的index，
#   然后遍历end 直到直到第一个为奇数的index 然后交换
#   直到两者相邻
#   如果要考虑抽象的话 就需要将判定函数抽象出来,这样就可以扩展
#   不同规则的排列 比如负数与非负数 能被3整除的和不能被3整除的

def swap_num(L=[]):
    if L is None or len(L) == 0:
        return L
    startIndex = 0
    endIndex = len(L) - 1

    while startIndex < endIndex:

        if endIndex - startIndex == 1:
            # 说明 已经遍历完成了 就判定是否要进行交互
            if L[endIndex] // 2 == 0 and L[endIndex] // 2 != 0:
                temple = L[startIndex]
                L[startIndex] = L[endIndex]
                L[endIndex] = temple
            break
        if not isEven(L[startIndex]):
            # 说明开始的标志是奇数 那么 往后移动
            startIndex += 1
        elif isEven(L[endIndex]):
            # 说明后面的标志数偶数 那么往前移动
            endIndex -= 1
        else:
            # 说明开始的标志数偶数 并且结束的标志数奇数
            temple = L[startIndex]
            L[startIndex] = L[endIndex]
            L[endIndex] = temple
            startIndex += 1
            endIndex -= 1

    return L


# 判定函数 是否是偶数
def isEven(n):
    return n & 0x1 == 0


# print(swap_num([1, 3, 4, 2, 6, 7, 9]))

# -------------------------------------------------------
#
# 11、链表中倒数第k个结点（Page：124）
# 输入一个链表，输出该链表中倒数第k个结点。
# 比如链表1、2、3、4 的倒数第一个结点就是4
# 思路：
# 输入的是开始节点 正常思路是 从头开始 遍历到尾部
# 再回溯打印倒数 但是 这是个单向链表
# 所以无法回溯
# 先遍历 将链表逆向 再打印（好像是可行的）
# 有没有更好的方式呢
# 我们总共有4个节点 那么倒数第1个节点 就是末尾节点
# 倒数第3个节点 就是节点3
# 我们可以定义两个指针，start end
# start先走k步 当start走到了末尾节点的时候
# start = end + k -1
# print_index = end
# 比如 打印倒数第3个节点 2
# start = 1 end = 0
# start = 2  end = 0
# start = 3  end = 1
# start = 4  end = 2

def print_k(startNode, k):
    if startNode is None:
        print('输入的链表有误 ')
        return
    if k == 0:
        print('输入的节点数有误 ')
        return
    start_index = startNode

    i = 0
    flag = 1
    while i < k - 1:
        start_index = start_index.nextNode
        if start_index is None:
            print('输入的节点数 大于 链表的节点数 error')
            flag = 0
            break
        i += 1
    if not flag:
        return
    end_index = startNode
    while start_index is not None and start_index.nextNode is not None:
        start_index = start_index.nextNode
        end_index = end_index.nextNode

    print(end_index.value)


# E = ListNode(5, None)
# D = ListNode(4, E)
# C = ListNode(3, D)
# B = ListNode(2, C)
# A = ListNode(1, B)
#
# print_k(A, 5)
# print_k(B, 5)
# print_k(A, 3)
# print_k(A, 2)
# print_k(E, 1)

# -------------------------------------------------------
#
# 12、反转链表（Page：129）
# 定义一个函数，输入一个链表的头结点，反转该链表并输出反转后链表的头结点
# 思路：
#   用一个节点储存当前节点，
#   last current next
#

#   last = current
#   current = next
#   next = next.next

def reverse_list(startNode):
    if startNode is None:
        print(" 头节点是None哟")
        return
    if startNode.nextNode is None:
        print("只有一个节点 不需要反转")
        return
    currentNode = startNode
    nextNode = startNode.nextNode
    lastNode = None
    while nextNode is not None:
        # 将cur的next指向last
        currentNode.nextNode = lastNode

        # 往下移动一位
        lastNode = currentNode
        currentNode = nextNode
        nextNode = nextNode.nextNode

        print(currentNode)
        if nextNode is not None:
            print(' last:%s  cur:%s  next:%s ' % (lastNode.value, currentNode.value, nextNode.value))
        else:
            currentNode.nextNode = lastNode
    return currentNode


# E = ListNode(5, None)
# D = ListNode(4, E)
# C = ListNode(3, D)
# B = ListNode(2, C)
# A = ListNode(1, B)
#
# start = reverse_list(E)
# print(start)
# while start is not None:
#     print(start.value)
#     start = start.nextNode

# -------------------------------------------------------
#
# 13、合并两个排序的链表（Page：131）
# 输入两个递增排序的链表，合并这两个链表并使得新链表中的结点仍然是按照递增排序的。例如下图
# 链表1  1->3->5->7
# 链表2  2->4->6->8
# 链表3  1->2->3->4->5->6->7->
#
# 思路：
#   不多说 肯定是依次放入两个链表 然后移动指针

def merge_list(list_a, list_b):
    if list_a is None and list_b is None:
        print('两个链表都是None哟')
        return
    if list_b is None:
        return list_a
    if list_a is None:
        return list_b

    # 新建一个列表
    if list_a.value > list_b.value:
        next_list = list_b
        list_b = list_b.nextNode
    else:
        next_list = list_a
        list_a = list_a.nextNode
    headNode = next_list
    while list_a is not None or list_b is not None:
        # 当两者都为None的时候 就结束
        if list_b is None:
            next_list.nextNode = list_a
            list_a = list_a.nextNode
        if list_a is None:
            next_list.nextNode = list_b
            list_b = list_b.nextNode

        if list_a is not None and list_b is not None:
            if list_a.value < list_b.value:
                next_list.nextNode = list_a
                list_a = list_a.nextNode
            else:
                next_list.nextNode = list_b
                list_b = list_b.nextNode

        next_list = next_list.nextNode
    return headNode


# check code
# G = ListNode(7, None)
# E = ListNode(5, G)
# C = ListNode(3, E)
# A = ListNode(1, C)
#
# H = ListNode(8, None)
# F = ListNode(6, H)
# D = ListNode(4, F)
# B = ListNode(2, D)
#
# resultNode = merge_list(A, B)
# while resultNode is not None:
#     print(resultNode.value)
#     resultNode = resultNode.nextNode


# -------------------------------------------------------
#
# 14、树的子结构（Page：134）
# 输入两颗二叉树A和B，判断B是不是A的子结构，二叉树的结点的定义如下：
# class TreeNode{
#   int value;
#   TreeNode left_node;
#   TreeNode right_node;
# }
#   树 A             树 B
#        8                 8
#    8       7          9     2
# 9     2
#    4     7
#  如上 右边的树B 就是左边的树A的子树
#
# 思路：
#   没有说树是排过序的
#   所以 用递归的方式 输入两个树的起始节点 判定 所有的子树的子节点是否一致
#   如果value一致的话  就hasTree1InTree2
#   如果没有的话 就判定左树
#   如果还是没有 就判定右树
#
#   has_tree1_in_tree2函数
#   说明当前节点的value一致 判断子节点是否一致
#   广西百花香果

def has_sub_tree(tree_a, tree_b):
    result = False
    if tree_a is not None and tree_b is not None:
        if tree_a.value == tree_b.value:
            result = has_tree1_in_tree2(tree_a, tree_b)
        if not result:
            result = has_sub_tree(tree_a.leftNode, tree_b)

        if not result:
            result = has_sub_tree(tree_a.rightNode, tree_b)

    return result


def has_tree1_in_tree2(tree_a, tree_b):
    if tree_a is None:
        return False
    if tree_b is None:
        return True
    if tree_a.value != tree_b.value:
        # 如果两个节点的value不一致 那么就false
        return False
    # 说明两个节点的value一致 那么判定他们的左右节点是否一致
    return has_tree1_in_tree2(tree_a.rightNode, tree_b.rightNode) and has_tree1_in_tree2(tree_a.leftNode,
                                                                                         tree_b.leftNode)


four4 = TreeNode(4, None, None)
four7 = TreeNode(7, None, None)
three9 = TreeNode(9, None, None)
three2 = TreeNode(2, four4, four7)
two8 = TreeNode(8, three9, three2)
two7 = TreeNode(7, None, None)
one8 = TreeNode(8, two8, two7)

child9 = TreeNode(9, None, None)
child2 = TreeNode(2, None, None)
child8 = TreeNode(8, child9, child2)

print(has_sub_tree(one8, child8))
