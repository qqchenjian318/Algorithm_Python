# coding=utf-8

# 剑指offer的算法题 python实现  18到30题


# 链表的节点
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


# ---------------------------------------------------------------
#
# 18、栈的压入、弹出序列（P：151）
# 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
# 假设压入栈的所有数字均不相等。例如序列1、2、3、4、5是某栈的压栈序列，
# 序列4、5、3、2、1是该压栈序列对应的一个弹出序列，但4、3、5、1、2就不可能是该压栈序列的弹出序列。
# （压栈并不代表会一下子全部压进入，可能是先压入1、2、3、4，然后弹出4，接着压入5
# ，然后弹出，5、3、2、1，所以序列4、5、3、2、1才属于压栈序列1、2、3、4、5的一个弹出序列）
# 栈 先进后出
# 思路：
#   建立一个辅助栈，
#   先看，弹出序列 需要是的4 ，辅助栈是空的 那么先将压入栈中的数字压入辅助栈 直到4在辅助栈的栈顶
#   目前的情况是 辅助栈  1、2、3、4   压入序列 5    弹出序列    4、5、3、2、1
#   然后弹出辅助栈的4 和弹出序列的4
#   现在弹出序列需要的是5 ，看辅助栈的栈顶是否是5，当前是3不是5
#   那么继续在压入序列中压入数字，直到取到5
#   此时的辅助栈 1、2、3、5   压入序列 空     弹出序列    5、3、2、1
#   弹出辅助栈和弹出序列的5
#   此时 辅助栈1、2、3     压入序列 空  弹出序列 3、2、1
#   依次判断辅助栈的栈顶是否跟弹出序列一致

def stack_sequence(inStack=[], outStack=[]):
    # 排除异常情况
    if inStack is None and outStack is None:
        return True
    if inStack is None or outStack is None:
        return False
    if len(inStack) != len(outStack):
        return False
    # 建立辅助栈
    helperStack = []
    while len(outStack) != 0:
        # 直到弹出序列为空
        outTop = outStack.pop(0)
        if len(helperStack) != 0:
            # 说明辅助栈中有数据
            helperTop = helperStack.pop()
            print('辅助栈 弹出顺序 %s' % helperTop)
        else:
            # 说明辅助栈中没有数据 那么依次将压入栈的序列 压入 直到 遇到了outTop
            inTop = move_top(outTop, inStack, helperStack)
            # 到这里 有三种情况 一 inTop是None 说明压入栈中没有 想要的数据了
            if inTop is None:
                print('压入序列中没有需要的 数字')
                return False
            # 弹出刚才添加到辅助栈中的数字
            helperTop = helperStack.pop()
            print('辅助栈 弹出顺序 %s' % helperTop)

        if helperTop is None:
            print('辅助栈的栈顶数字不对 ')
            return False
        if helperTop != outTop:
            # 说明当前的辅助栈 和弹出序列不一致
            # 从压入序列中压入
            inTop = move_top(outTop, inStack, helperStack)
            if inTop is None:
                print('压入序列中没有需要的 数字')
                return False
                # 弹出刚才添加到辅助栈中的数字
            helperTop = helperStack.pop()
            print('辅助栈 弹出顺序 %s' % helperTop)
        else:
            # 说明当前的辅助栈和弹出序列一致
            print('辅助栈和弹出序列一致 ')
    return True


# 从inStack中 压入数字 outTop == 该数字
def move_top(outTop, inStack, helperStack):
    inTop = None
    while len(inStack) != 0 and inTop != outTop:
        # 说明压入序列还有值 而且当去inTop != outTop
        inTop = inStack.pop()
        helperStack.append(inTop)
    return inTop


# inStack = [1, 2, 3, 4, 5]
# outStack = [4, 3, 5, 1, 2]
# print(stack_sequence(inStack, outStack))


# ---------------------------------------------------------------
#
# 19、从上往下打印二叉树（P：154）
# 从上往下打印出二叉树的每个结点，同一层的节点按照从左到右的顺序打印，
# 例如输入下图中的二叉树，则依次打印出8、6、10、5、7、9、11。
#         8
#     6      10
#   5   7  9   11
# 思路：
#   递归打印每一层的数据
#

def print_tree(topTree):
    if topTree is not None:
        print(topTree.value)

    if topTree.leftNode is not None:
        print_tree(topTree.leftNode)
    if topTree.rightNode is not None:
        print_tree(topTree.rightNode)


a = 0.12
b = 0.11
c = a + b
print(c)
