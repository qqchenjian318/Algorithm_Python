# coding=utf-8

# 剑指offer的算法题 python实现  18到30题


# 链表的节点
class ListNode(object):
    def __init__(self, value=None, nextNode=None):
        self.value = value
        self.nextNode = nextNode


class SliListNode(object):
    def __init__(self, value=None, nextNode=None, sibling=None):
        self.value = value
        self.nextNode = nextNode
        self.sibling = sibling


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

def print_tree(topTree, level=0):
    if topTree is not None and level == 0:
        print(topTree.value)
    level += 1
    if topTree.leftNode is not None:
        print(topTree.leftNode.value)
    if topTree.rightNode is not None:
        print(topTree.rightNode.value)

    if topTree.leftNode is not None:
        print_tree(topTree.leftNode, level)
    if topTree.rightNode is not None:
        print_tree(topTree.rightNode, level)


# three_5 = TreeNode(5)
# three_7 = TreeNode(7)
# three_9 = TreeNode(9)
# three_11 = TreeNode(11)
# two_6 = TreeNode(6, three_5, three_7)
# two_10 = TreeNode(10, three_9, three_11)
# one_8 = TreeNode(8, two_6, two_10)
#
# print_tree(one_8)

# ---------------------------------------------------------------
#
# 20、二叉搜索树的后序遍历序列（P：157）
# 输入一个整数数组，判断该数组是不是某二叉搜索树的后续遍历的结果。如果是则返回true，否则返回false。
# 假设输入的数组的任意两个数字都互不相同。比如数组5、7、6、9、11、10、8就是一个二叉树的后序遍历序列，而数组7、4、6、5不是
# （后序遍历序列：先访问左子节点，再访问右子节点，最后访问根节点）
# 思考：
#   根据后序遍历的规则 最后一次一个值 肯定是二叉树的根根节点
#   1、在根据规则 遍历 数组 找到分界线 小于根节点的 都是左树 大于根节点的都是右树
#   2、如果第一个值 小于 最后一个值 那么二叉树 拥有左树
#   3、如果第一个值 大于 最后一个值 那么二叉树 没有子树

def lrd_tree(array):
    if array is None or len(array) <= 0:
        return False
    length = len(array)
    topTree = array[length - 1]
    where = 0
    for i in range(length):
        if array[i] > topTree:
            where = i
            break
    # 找到了中间线 那么左边 是一棵树 右边也是一颗树
    # 右边的都大于根节点
    startIndex = where
    while startIndex < length:
        if array[startIndex] < topTree:
            return False
        startIndex += 1
    # 判断左树是否满足要求
    leftTree = True
    rightTree = True
    if where > 0:
        leftTree = lrd_tree(array[0:where])
    if where < length - 1:
        rightTree = lrd_tree(array[where:length - 1])

    return leftTree and rightTree


#
# firstArray = [5, 7, 6, 9, 11, 10, 8]
# secondArray = [7, 4, 6, 5]
# print(lrd_tree(firstArray))
# print(lrd_tree(secondArray))

# ---------------------------------------------------------------
#
# 31、二叉树中和为某一值的路径（P：160）
# 输入一个二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。
# 从树的根节点开始往下一直到叶节点所经过的节点行程一条路径。
# （使用前序遍历，先根节点，然后左，最后右）
# 思路：
#     其实就是使用前序遍历，打印出二叉树的所有路径，
#     因为对路径是有要求的，所以 进行符合要求的遍历即可
#          8
#       2     10
#     1   3 9    11
#    需要使用一个stack 保存当前的路径
#   比如 8 2 1 返回当前的时候 退出节点
#   叠加节点之后 计算值是否满足条件 如果满足 直接弹出值
#   如果小于 则进行 下一个节点的判定

stack = []


def frd_tree(topTree, lastValue, toValue):
    if topTree is None:
        return
    stack.append(topTree.value)
    lastValue += topTree.value
    if lastValue == toValue:
        print(stack)
    elif lastValue > toValue:
        pass
    else:
        # 说明小于 那么进先找左边的
        if topTree.leftNode is not None:
            frd_tree(topTree.leftNode, lastValue, toValue)
        # 再找右边的
        if topTree.rightNode is not None:
            frd_tree(topTree.rightNode, lastValue, toValue)
    lastValue -= toValue
    stack.pop()


# three_4 = TreeNode(4)
# three_7 = TreeNode(7)
# two_5 = TreeNode(5, three_4, three_7)
# two_12 = TreeNode(12)
# one_10 = TreeNode(10, two_5, two_12)
# frd_tree(one_10, 0, 22)

# ---------------------------------------------------------------
#
# 22、复杂链表的复制（P：164）
# 请实现一个函数，复制一个复杂链表，在复杂链表中，
# 每个结点除了有一个next指针指向下一个节点外，
# 还有一个sibling指针指向链表中的任意节点或者NULL。
# 如下图，直线表示的是next指针，而虚线表示的是sibling指针
#
# 思考
#   其实复杂的是sibling指针
#   思路一：先复制正常的所有节点，然后再遍历原链表 复制sibling指针
#           这样的话 每个sibling都需要遍历一次链表，才能找到真实的位置在哪里，所以时间复杂度是O（n^2）
#   思路二：有没有什么方式，可以把时间复杂度降低到O（n）呢？
#           应该是可以的，比如，首先还是复制基础链表，并且创建一个Hash表，将当前value和该节点的地址对应起来
#           这样，在复制每个节点的sibling指针的时候，就可以根据指向的value，在Hash表里面查询对应的地址，然后赋值
#           这样可以把每个节点的Sibling指针的时间复杂度降低到O(1)。这样其实就是用时间换空间的思想
#   思路三：有没有什么方式可以在不利于辅助空间的情况下，将时间复杂度降低到O(n)呢？
#           每个节点复制的时候，创建一个复制节点—》A—》A'—》B—》B’—》C—》C'
#           然后每个复制节点的s指针，指向的就是原节点的s指针的nextNode。
#           最后，取所有偶数节点出来 就是新的复制成功的链表
def copy_list(firstNode):
    if firstNode is None or firstNode.nextNode is None:
        print("链表返回了哟")
        return
        # 第一步创建新的复制链表
    tempNode = firstNode
    while tempNode.nextNode is not None:
        # 如果下个节点不为None
        copyNode = SliListNode(tempNode.value, tempNode.nextNode)
        tempNode.nextNode = copyNode
        tempNode = copyNode.nextNode
    lastNode = SliListNode(tempNode.value, tempNode.nextNode)
    tempNode.nextNode = lastNode
    # 复制了基础链表 下一步是复制s指针
    tempS = firstNode
    while tempS is not None:
        if tempS.sibling is not None:
            # 如果当前节点有s指针，那么下一个节点的s指针就指向该节点的s指针的下一个节点
            tempS.nextNode.sibling = tempS.sibling.nextNode
        tempS = tempS.nextNode.nextNode

    # 从头遍历链表 取出偶数节点 组成链表
    resultNode = firstNode.nextNode
    while firstNode is not None and firstNode.nextNode is not None:
        firstNode.nextNode = firstNode.nextNode.nextNode
        firstNode = firstNode.nextNode
    return resultNode


# s_5 = SliListNode(5)
# s_4 = SliListNode(4, s_5)
# s_3 = SliListNode(3, s_4)
# s_2 = SliListNode(2, s_3)
# s_1 = SliListNode(1, s_2)
#
# s_1.sibling = s_3
# s_2.sibling = s_5
# s_4.sibling = s_2
#
# result = copy_list(s_1)
#
# while result is not None:
#     if result.sibling is not None:
#         print(result.value, result.nextNode, result.sibling.value)
#     else:
#         print(result.value, result.nextNode, result.sibling)
#     result = result.nextNode

# ---------------------------------------------------------------
#
# 23、二叉搜索树与双向链表（P：168）
# 输入一个二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
# 要求不能创建任何新的节点，只能调整树中节点指针的指向。如下图
#
#       10
#   6       14     >>>> 4《==》6《==》8《==》10《==》12《==》14《==》16
# 4   8   12   16
#
# 思路：
#   因为要求链表是排好序的，所以就是使用二叉树的中序搜索 ，（先左 再父 后右）
#   但是树节点只有左右指针
#   所以我们定义右指针 指向下一个节点 左边指针指向上一个节点？
#   其实问题可以分解成，左树 转换 右树转换 最后连接起来
#   返回最后一个节点
#   左边树 返回最大值 右边树 返回最小值 整体返回最大值

def tree_2_list(topTree):
    pass
    # if topTree is None:
    #     return
    # resultStart = topTree
    # resultEnd = topTree
    # if topTree.leftNode is not None:
    #     leftNodes = tree_2_list(topTree.leftNode)
    #     leftNodes[1].rightNode = topTree
    #     topTree.rightNode = leftNodes[1]
    #     resultStart = topTree
    #
    # if topTree.rightNode is not None:
    #     rightNodes = tree_2_list(topTree.rightNode)
    #     rightNodes[0].leftNode = topTree
    #     topTree.rightNode = rightNodes[0]
    #
    #     resultEnd = rightNodes[1]
    # return resultStart, resultEnd

# 这个题没写完有问题
# one_4 = TreeNode(4)
# one_8 = TreeNode(8)
# one_12 = TreeNode(12)
# one_16 = TreeNode(16)
# two_6 = TreeNode(6, one_4, one_8)
# two_14 = TreeNode(14, one_12, one_16)
# three_10 = TreeNode(10, two_6, two_14)
#
# rightList = tree_2_list(three_10)[0]
#
# while rightList is not None:
#     print(rightList.value)
#     rightList = rightList.rightNode

# ---------------------------------------------------------------
#
# 24、字符串的排列（P：171）
# 输入一个字符串，打印出该字符串中字符的所有排列。
# 例如输入字符串abc，则打印出由字符a、b、c所能排列出来的所有字符串abc、acb、bac、bca、cab和cba
# 思路：
#   数学上叫排列组合
#   字符串先分解成字符数组，然后startIndex依次和后面的交换 交换之后 、
#   再对后面的进行交换计算，其实就是递归操作
#   a bc   bca  bac
#   c ba  cab
#   a cb
#   设定输入的是一个字符数组

def print_str(strArray):
    if strArray is None or len(strArray):
        print("input array is None")
        return


