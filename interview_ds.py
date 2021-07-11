#coding: utf-8


# 解题步骤：
# 1. constraints，数组大小、输入是否为unicode、robot是否可以在迷宫斜对角行走
#     找出所有的限制条件，subsequence是否连续
# 2. ideas，找出可能解决的办法
#     > 简化问题，二维变一维
#     > 尝试举例，发现pattern
#     > 思考合适的数据结构
# 3. complexity，计算时空复杂度
# 4. write code，在写之前，跟面试官商量好思路的可行性
# 5. test，非常重要，经常会被忽略，拼写、边界等


class solution1(object):
    def insertIntervals(self, intervals, newInterval):
        # insert intervals
        res = []
        i = 0
        while intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1
        if newInterval[1] < intervals[i][0]:
            res.append(newInterval)
        else:
            start = min(intervals[i][0], newInterval[0])
            end = start

            while intervals[i][0] <= newInterval[1]:
                end = max(intervals[i][1], newInterval[1])
                i += 1
            res.append([start, end])

        while i < len(intervals):
            res.append(intervals[i])
            i += 1
        return res
# sol = solution1()
# a=[[1,2],[3,4],[6,7],[9,10],[12,16]]
# b=[5,9]
# print sol.insertIntervals(a, b)

from operator import itemgetter
class solution2(object):
    def mergeIntervals(self, intervals):
        # merge intervals
        intervals.sort(key=itemgetter(0, 1))
        res = []
        start, end = intervals[0]
        idx = 1
        itr = 0
        while idx < len(intervals):
            while end >= intervals[idx][0]:
                end = max(intervals[idx][1], end)
                idx += 1
            res.append([start, end])
            if idx == len(intervals)-1:
                res.append(intervals[-1])
                break
            elif idx == len(intervals):
                break
            else:
                start, end = intervals[idx]
                idx += 1
        return res
# sol = solution2()
# a=[[1,7],[3,4],[6,7],[3,10],[12,16]]
# print sol.mergeIntervals(a)


## meeting rooms I
class solution3(object):
    def canAttendMeetings(self, intervals):
        intervals.sort(key=lambda r:r[0])
        for idx in range(len(intervals)-1):
            if intervals[idx][1] > intervals[idx+1][0]:
                return False
        return True
array = [[0, 6],[15, 20], [5, 10]]
# sol = solution3()
# print sol.canAttendMeetings(array)


## meeting rooms II
from collections import OrderedDict
class solution4(object):
    def minMeetingRooms(self, intervals):
        m = dict()
        for itv in intervals:
            if itv[0] not in m:
                m[itv[0]] = 1
            else:
                m[itv[0]] += 1
            if itv[1] not in m:
                m[itv[1]] = -1
            else:
                m[itv[1]] -= 1
        # 找到一个数据结构，key为起始时间，value为起始时间+1，终点时间-1
        # list(下标是时间)：如果起始时间很大的话可能导致list很长，浪费空间
        # dict：最多存储2n个元素，再排序，复杂度为O(nlgn)
        # heap：排序的元素为起始时间，nlgn
        m = sorted(m.items(), key=lambda r:r[0])
        min_rooms, num = 0, 0
        for item in m:
            num += item[1]
            min_rooms = max(min_rooms, num)
        return min_rooms
# sol = solution4()
# a=[[0,30],[5,10],[10,20]]
# print sol.minMeetingRooms(a)


## split interval, overlap 部分, weight 相加，解题方法类似meeting rooms II
## 比如, 【1,3】, 【2,4】, weight 各为 0.5 输出,【1,2】,【2,3】,【3,4】 weight 各为 0.5, 1, 0.5
class solution4_1(object):
    def splitInterval(self, array, weight):
        m = {}
        for i, a in enumerate(array):
            if a[0] not in m:
                m[a[0]] = weight[i]
            else:
                m[a[0]] += weight[i]
            if a[1] not in m:
                m[a[1]] = -weight[i]
            else:
                m[a[1]] -= weight[i]
        m = sorted(m.items(), key=lambda r:r[0])
        new_array = []
        new_weight = []
        total = 0
        for i in range(len(m)-1):
            new_array.append([m[i][0], m[i+1][0]])
            total += m[i][1]
            new_weight.append(total)
        return new_array, new_weight
# sol=solution4_1()
# a,w=[[1,3],[2,4]],[0.5,0.5]
# a,w=[[0,30],[5,10],[10,20]],[0.5,1,0.5]
# print sol.splitInterval(a,w)


# Median of two sorted arrays
class solution5(object):
    def MedianOfTwoSortedArrays(self, arrayA, arrayB):
        halfA = len(arrayA)/2
        halfB = len(arrayB)/2
        kth = halfA+len(arrayA)%2+halfB+len(arrayB)%2

        return self.recursive(arrayA, arrayB, kth)

    def recursive(self, arrayA, arrayB, kth):
        if kth == 1:
            return min(arrayA[0], arrayB[0])
        if len(arrayA) == 0:
            return arrayB[kth-1]
        if len(arrayB) == 0:
            return arrayA[kth-1]
        halfA = len(arrayA)/2
        halfB = len(arrayB)/2

        midA = (arrayA[halfA-1] + arrayA[halfA])/2. if len(arrayA)%2==0 else arrayA[halfA]        
        midB = (arrayB[halfB-1] + arrayB[halfB])/2. if len(arrayB)%2==0 else arrayB[halfB]

        if midA < midB:
            return self.recursive(arrayA[halfA:], arrayB, kth-halfA)
        elif midA > midB:
            return self.recursive(arrayA, arrayB[halfB:], kth-halfB)
        else:
            return midA
# sol = solution5()
# A=[1,2,3]
# B=[4,5]
# print sol.MedianOfTwoSortedArrays(A, B)


# Subarray Sum I
class Solution6:
    def subarraySum(self, nums):
        dict = {}
        dict[0] = -1
        sum = 0
        res = []
        for i, num in enumerate(nums):
            sum += num
            if sum in dict:
                res.append(dict[sum] + 1)
                res.append(i)
                return res
            dict[sum] = i
# sol = Solution6()
# a = [-3,2,1,3,5]
# print sol.subarraySum(a)


# Subarray Sum Closest
class solution7(object):
    def subarraySum(self, nums):
        sum = 0
        sum_array = []
        for idx, num in enumerate(nums):
            sum += num
            sum_array.append([sum, idx])

        sum_array.sort(key=lambda r:r[0])

        left, right = 0, 0
        sum_closest = 2^31-1
        for idx in range(len(sum_array)-1):
            tmp = abs(sum_array[idx][0] - sum_array[idx+1][0])
            if tmp < sum_closest:
                sum_closest = tmp

                left = sum_array[idx+1][1]+1 if sum_array[idx][1] > sum_array[idx+1][1] else sum_array[idx][1]+1
                right = sum_array[idx+1][1] if sum_array[idx][1] < sum_array[idx+1][1] else sum_array[idx][1]

        return left, right
# sol = solution7()
# a = [-4, 2, 4, -5, 6]
# print sol.subarraySum(a)


# Hash function
def hash_func(string):
    sum = 0
    for s in string:
        sum = sum*31 + ord(s)
        sum = sum % HASH_TABLE_SIZE
    return sum
# print hash_func('sf')


# Permutation
class solution8(object):
    def permutation(self, nums):
        self.nums = nums
        self.res = []
        visited = set()
        permut = []
        self.find(permut, visited)
        return self.res

    def find(self, permut, visited):
        if len(permut) == len(self.nums):
            self.res.append(permut)
            return

        for i in self.nums:
            if i in visited:
                continue

            permut.append(i)
            visited.add(i)
            self.find(permut, visited)
            visited.remove(i)
            permut.pop()
# sol = solution8()
# a=[1,2,3]
# print sol.permutation(a)


# sliding puzzle
from Queue import Queue
class solution9(object):
    def slidePuzzle(self, nums):
        queue = Queue()
        state_visited = set()

        directions = [[1,3],[0,2,4],[1,5],[0,4],[1,3,5],[2,4]]
        initial_state = ''.join([''.join([str(n) for n in num]) for num in nums])
        queue.put(initial_state)

        res = 0
        while queue.qsize() > 0:
            for _ in range(queue.qsize()):
                state = queue.get()

                if state == '123450':
                    return res

                zero_index = state.index('0')
                for direct in directions[zero_index]:
                    tmp_state = self.swap(state, zero_index, direct)
                    print zero_index, direct, state, tmp_state

                    if tmp_state in state_visited:
                        continue

                    state_visited.add(tmp_state)
                    queue.put(tmp_state)

            res += 1

        return -1

    def swap(self, s, i, j):
        lst = list(s)
        lst[i], lst[j] = lst[j], lst[i]
        return ''.join(lst)
# sol = solution9()
# a=[[1,2,3],[4,0,5]] # 1
# b=[[1,2,3],[5,4,0]] # -1
# c=[[4,1,2],[5,0,3]] # 5
# d=[[3,2,4],[1,5,0]] # 14
# print sol.slidePuzzle(d)


# word break II
class Solution10(object):
	def wordBreakTwo(self, source, words):
		self.m = dict()
		return self.dfs(source, words)

	def dfs(self, source, words):
		if source == '':
			return ['']
		if self.m.get(source):
			return self.m[source]

		result = []
		for word in words:
			if source[:len(word)] == word:
				res = self.dfs(source[len(word):], words)
				for r in res:
					result.append(word + (' ' + r if r else ''))
		self.m[word] = result
		return result
# sol = Solution10()
# s='catsanddog'
# words=["cat", "cats", "and", "sand", "dog"]
# print sol.wordBreakTwo(s, words)


# word search I
class Solution11(object):
	def wordSearch(self, board, word):
		for i in range(len(board)):
			for j in range(len(board[0])):
				if board[i][j] == word[0] and self.search(board, i, j, word, i, j):
					return True
		return False

	def search(self, board, i, j, word, prev_i, prev_j):
		if len(word) == 0:
			return True
		m, n = len(board), len(board[0])
		if i < 0 or i >= m or j < 0 or j >= n:
			return False
		if board[i][j] == word[0]:
			for d in [[0, -1], [0, 1], [1, 0], [-1, 0]]:
				if i+d[0] == prev_i and j+d[1] == prev_j:
					continue
				if self.search(board, i+d[0], j+d[1], word[1:], i, j):
					return True
		return False
# sol = Solution11()
# board=['ABCE','SFCS','ADEE']
# print sol.wordSearch(board, 'ABCCED')
# print sol.wordSearch(board, 'SEE')
# print sol.wordSearch(board, 'ABCB')


# shortest Distance from All Buildings
from Queue import Queue
class Solution12(object):
	def ShortestDistance(self, grids):
		m, n = len(grids), len(grids[0])

		buildings = []
		for i in range(m):
			for j in range(n):
				if grids[i][j] == 1:
					buildings.append([i, j])

		result = [[-1]*n]*m
		for build in buildings:
			dist = 0
			dists = [[-1]*n]*m
			queue = Queue()
			queue.put(build)
			dists[i][j] = dist
			while queue.qsize() > 0:
				dist += 1
				for _ in queue.size():
					cur_pos = queue.get()
					for d in [[0, -1], [0, 1], [1, 0], [-1, 0]]:
						i, j = cur_pos[0]+d[0], cur_pos[1]+d[1]
						if i < 0 or i >= m or j < 0 or j >= n:
							continue
						if grids[i][j] != 0:
							continue
						if dists[i][j] != -1:
							continue
						queue.put([i, j])
						dists[i][j] = dist
			for i in range(m):
				for j in range(n):
					if dists[i][j] != -1:
						result[i][j] += dists[i][j]
		shortest_path = -1
		for i in range(m):
			for j in range(n):
				if result[i][j] != -1:
					shortest_path = min(shortest_path, result[i][j])
		return shortest_path



class Solution13(object):
    def validPalindrome(self, array):
        # valid palindrome II
        return self.valid(array, 0)

    def valid(self, array, count):
        if len(array) <= 1:
            return True
        if array[0] == array[-1]:
            if self.valid(array[1:-1], count):
                return True
        else:
            if count == 1:
                return False
            if self.valid(array[1:], 1) or self.valid(array[:-1], 1):
                return True
        return False


    def longestPalindrome(self, s):
        """
        longest palindrome substring
        Manacher算法: 待理解！！！
        step 1: 字符串内插入特殊字符'#'，处理后字符串长度为奇数；字符串收尾插入特殊字符，避免数组越界
        step 2:逐个遍历字符，计算得到以每个字符为中心的最长回文串半径。
        涉及到的变量有：存储字符i回文半径的数组P，上一个回文串的中心位置c以及回文串结束位置r。
        计算字符i回文半径：本次计算尽量利用之前回文串匹配的结果，减少重复字符比对。
        """
        if not s:
            return None
        if len(s)<2:
            return s
        T='#'.join('@{}$'.format(s))
        # T: @#a#b#c#c#b#d#$
        n=len(T)
        P=[0]*n
        c=0
        r=0
        for i in range(1,n-1):
            #i关于中心c的对称位置
            i_mirror=2*c-i
            print T[i], c, r, P[i]
            #利用之前回文串字符对比重复部分
            if r>i:
                P[i]=min(r-i, P[i_mirror])
            # 中心扩展法完成之前没有涉及的字符比对
            while T[i+1+P[i]]==T[i-1-P[i]]:
                P[i]=P[i]+1
            #更新当前回文串中心c及终止位置r
            if i+P[i]>r:
                c=i
                r=i+P[i]
        #找到最大回文半径及对应的回文中心
        maxlen=0
        centeridx=0
        for i in range(1,n-1):
            if P[i]>maxlen:
                maxlen=P[i]
                centeridx=i
        #获取最长回文串
        begin=(centeridx-maxlen)/2
        end=(centeridx+maxlen)/2
        return s[begin:end]
# sol = Solution13()
# a='jss'
# print sol.validPalindrome(a)
# a='abccbd'
# print sol.longestPalindrome(a)


# deepest leaf LCA
class Node(object):
    def __init__(self, value):
        self.val = value
        self.path = 0
        self.children = []

class Solution14(object):
    def LCA(self, root):
        res = self.dfs(root)
        return res.val

    def generateTree(self):
        root = Node(1)
        node2 = Node(2)
        node3 = Node(3)
        node4 = Node(4)
        node5 = Node(5)
        node6 = Node(6)
        node7 = Node(7)
        node8 = Node(8)
        node9 = Node(9)
        node10 = Node(10)
        root.children.extend([node2,node3])
        node2.children.extend([node4,node5,node6])
        node3.children.extend([node7])
        # node4.children.extend([node8])
        # node5.children.extend([node9])
        # node8.children.extend([node10])
        return root

    def dfs(self, root):
        if root and len(root.children) == 0:
            return root
        max_deepth = 0
        max_deep_count = 0
        max_deep_node = None
        for child in root.children:
            node = self.dfs(child)
            if node.path > max_deepth:
                max_deepth = node.path
                max_deep_count = 1
                max_deep_node = node
            elif node.path == max_deepth:
                max_deep_count += 1
                max_deep_node = node
        if max_deep_count == 1 and max_deepth != 0:
            max_deep_node.path = 1 + max_deepth
            return max_deep_node
        else:
            root.path = 1 + max_deepth
            return root
        print 'Error'
# sol = Solution14()
# root = sol.generateTree()
# print sol.LCA(root)


# build balanced BST
class Node(object):
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

class Solution15(object):
    def buildBalancedBST(self, array):
        return self.recursive(array, 0, len(array)-1)

    def printTree(self, root):
        if not root:
            return
        self.printTree(root.left)
        print root.val
        self.printTree(root.right)

    def recursive(self, array, start, end):
        if start == end:
            return Node(array[start])
        if start == end - 1:
            tmp = Node(array[start])
            tmp.right = Node(array[end])
            return tmp
        mid = start + (end - start)/2
        node = Node(array[mid])
        node.left = self.recursive(array, start, mid-1)
        node.right = self.recursive(array, mid+1, end)
        return node
# sol = Solution15()
# root = sol.buildBalancedBST(range(9))
# sol.printTree(root)


# max heap
class MaxHeap(object):
    def __init__(self, capacity):
        self.data = []
        self.capacity = capacity
        self.count = 0

    def size():
        return self.count

    def isEmpty(self):
        return self.count == 0

    def getMax(self):
        if self.count > 0:
            return self.data[0]
        else:
            ValueError("Maxheap is empty, can't get max.")

    def push(self, item):
        if self.count >= self.capacity:
            raise ValueError("Maxheap is full, can't push.")
        self.data.append(item)
        self.count += 1
        self.siftUp(self.count-1)

    def pop(self):
        if self.count > 0:
            res = self.data[0]
            self.data[0] = self.data[self.count-1]
            self.count -= 1
            self.siftDown(0)
            return res
        else:
            raise ValueError("Maxheap is empty, can't pop.")

    def siftUp(self, index):
        while index > 0 and self.data[(index-1)/2] < self.data[index]:
            self.swap((index-1)/2, index)
            index = (index-1)/2

    def siftDown(self, index):
        while index*2+1 < self.count:
            i = index*2+1 
            if i+1 < self.count and self.data[i+1] > self.data[i]:
                i += 1
            if self.data[index] >= self.data[i]:
                break
            self.swap(i, index)
            index = i

    def swap(self, i, j):
        tmp = self.data[i]
        self.data[i] = self.data[j]
        self.data[j] = tmp
# sol = MaxHeap(10)
# sol.push(2)
# sol.push(5)
# sol.push(3)
# sol.push(1)
# print sol.getMax()
# print sol.pop()
# print sol.pop()
# print sol.pop()


# binary tree in-order iterator
from Queue import LifoQueue
class Node(object):
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

class Solution17(object):
    def __init__(self, node):
        self.stack = LifoQueue()
        while node:
            self.stack.put(node)
            node = node.left

    def hasNext(self):
        return self.stack.qsize() != 0

    def next(self):
        if self.hasNext():
            node = self.stack.get()
            res = node.val
            if node.right:
                node = node.right
                while node:
                    self.stack.put(node)
                    node = node.left
            return res
# root = Node(7)
# root.left = Node(3)
# root.right = Node(10)
# root.left.left = Node(1)
# root.left.right = Node(5)
# root.right.left = Node(8)
# sol = Solution17(root)
# print sol.hasNext()
# print sol.next()
# print sol.next()


# binary tree pre-order iterator
class Solution18(object):
    def __init__(self, node):
        self.stack = LifoQueue()
        self.stack.put(node)

    def hasNext(self):
        return self.stack.qsize() != 0

    def next(self):
        if self.hasNext():
            node = self.stack.get()
            if node.right:
                self.stack.put(node.right)
            if node.left:
                self.stack.put(node.left)
            return node.val
# root = Node(7)
# root.left = Node(3)
# root.right = Node(10)
# root.left.left = Node(1)
# root.left.right = Node(5)
# root.right.left = Node(8)
# sol = Solution18(root)
# print sol.hasNext()
# for _ in range(6): print sol.next()




