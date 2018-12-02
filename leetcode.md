# Leetcode


## 解题模版

### 小的Tip

* 规律1：见到需要维护一个集合的最小值/最大值的时候就要想到堆，例如第k小/大，可以是多个数组、也可以是矩阵
    - kth Smallest In m\*n Matrix: each column and row is sorted
    - kth Smallest In 3 Arrays: each array is unsorted
    - kth Smallest Sum In Two Sorted Arrays
    - Trapping Rain Water II

* 规律2：见到数组要先想到排序




### 二分法

* **说明**：
    - start+1 < end: 相邻即退出循环，永远不会死循环
    - mid = start + (end - start)/2 防止整数溢出
    - 简化条件判断，都是mid，可以不用考虑mid+1或者mid-1，当然考虑也是可以的
    - 最后只需要判断start、end两个位置的参数，顺序也可以相反
* **使用**：
    - 数组、字符串寻找起点终点等
    - 当前复杂度为O(n)，但是需要复杂度O(logn)
* **模版**：

```
void BS(vector<int> nums) {
    int start = 0；
    int end = nums.size()
    while (start+1 < end) {
        int mid = start + (end - start)/2;
        if (nums[mid] == target) {
            end/start = mid; // 根据题目意思来
        }
        else if (nums[mid] < target) {
            start = mid;
        }
        else {
            end = mid;
        }
    }

    if (nums[start] == target) {
        // 根据题目意思来
    }
    if (nums[end] == target) {
        // 根据题目意思来
    }
}
```

### 链表

* **说明**：
    - 使用dummy节点，即哑节点或者向导节点，只要返回的头节点发生了变化，
      一般都要用这个dummy node
    - 经常使用prev和cur两个节点来共同操作链表
    - 链表最大的特点是灵活，不用耗费额外空间，所以很多方法都可以达到空间复杂度要求
* **使用**：
    - 链表的插入、删除
    - 经常使用的技巧是[快慢指针](#双指针)
* **常见题型**：
    - Merge k Linked List，三种方法都要会（优先级队列、分治、两两合并）

* **模版**：
```
void LinkedList(node* head) {
    node* dummy = new node(0);
    node->next = head;

    node* prev = dummy;
    node* cur = head;
    while (head != NULL) {
        // to do
    }
}
```

### 数组和数字

* **常见题型**：
    - 排序数组
      融合两个排序数组、排序数组的交集、两个数组的点积
      两个有序数组的中点
    - 子数组
      购买股票
      子数组subarray
    - [双指针](#双指针)

### BFS和DFS

* **BFS模版**：
```
void BFS(int v) {
    visited[v] = true
    queue.push(v);
    while (!queue.empty())
        w = queue.pop();
        for (v的每一个邻接点w)
            if (!visited[w])
                visited[w] = true
                queue.pish(w)
}
```

### DFS

* **说明**：
    - 使用递归，非显示的使用栈，可以看出时复杂版的递归
    - 有的时候不一定需要判断是否访问过，即visited[v]不是必需，就看是否重复访问当前节点
    - 使用此种方式图一般使用临接表表示
* **使用**：
    - 一般寻找合适的路径、组合方式都适用图搜索
    - 隐式图：有的时候不一定是显示节点，比如棋盘、密码锁等，需要找出状态节点


* **DFS模版**：
```
void DFS(int v) {
    if (边界条件)
        return;
    visited[v] = true;
    for (v的每一个邻接点w) 
        if (!visited[w])
            DFS(w)
}
```

### 递归

* **说明**：
    - 可以看出，递归跟DFS很像，算是DFS的一种解法
    - 能够将问题拆分成非重叠子问题，便可以使用递归
    - 满足边界条件返回，最后合并返回结果
    - 当返回是所有的结果时，可以让函数返回为空，并在判断边界条件时收集符合条件的结果

* **常见题型**：
    - subsets I/II

* **模版**：
```
int recursive(int x, int y) {
    if (边界条件) 
        return;
    int a = recursive(x1, y1);
    int b = recursive(x2, y2);
    return a + b;
}
```

### 双指针

* **对撞型指针**：一个数组，从两边往中间移动
    - Two sum类
    - Partition类

Two Sum类:
```
if (array[i] + array[j] > target) {
    // do somthing;
    j--;
} else if (array[i] + array[j] < target) {
    // do something;
    i++;
} else {
    // do something;
    i++/j--;
}
```

Partition类：
```
int partition(int[] nums, int left, int right) {
    int pivot = nums[left];

    while (left < right) {
        while (left < right && nums[right] >= pivot) {
            right--;
        }
        nums[left] = nums[right];

        while (left < right && nums[left] <= pivot) {
            left++;
        }
        nums[right] == num[left];
    }

    // 返还pivot点到数组里面
    nums[left] = pivot;
    return left;
}

```

* **前向型指针**：一个数组，同时向前移动
    - 窗口类
    - 快慢类

窗口类，不同于滑动窗口
```
for (int i=0; i < n; i++) {
    while (int j < n) {
        if (condition) {
            j++;
            // 更新j的状态;
        } else {
            break;
        }
    }
    // 更新i状态
}

```

* **两个数组**：并行

两个数组分别使用一个指针，各找一个元素

* 常见题型：
    - 2 Sum
    - 2 Sum II
    - 3 Sum
    - 3 Sum Closest
    - 4 Sum
    - k Sum
    - Triangle Count：2 Sum类
    - Trapping Rain Water：2 Sum类
    - Container With Most Water：2 Sum类(待完成)
    - Sort colors：Partition类
    - Partition Array by Odd and Even：Partition类
    - kth Largest Element：Partition类
    - Valid Palindrome：Partition类
    - Sort Letters by Case：Partition类
    - Quict Sort/ Quick Select/ Nuts Bolts Problem/Wiggle Sort II：Partition类
    - Minimun Size Subarray Sum：窗口类
    - Longeset Substring Without Repeating Characters：窗口类
    - Longeset Substring wiht at Most k(two) Distinct Characters：窗口类
    - Mininmun Window Substring：窗口类
    - Find the Middle of Linked List：快慢类
    - Linked List Cycle I, II：快慢类
    - The Smallest Difference：并行类
    - Merge Two Sorted Lists：并行类

### 排序

* **快排和归并排序区别**：

* 区别：
    - 思路：快排是先整体有序，再局部有序；归并是先局部有序，再整体有序
    - 稳定性：快排不稳定；归并稳定排序
    - 时间复杂度：快排平均O(nlogn)，最坏是O(n^2)，归并最好最坏都是O(nlogn)
      空间复杂度：快排是O(1)，归并是数组是O(n)，链表是O(1)
* 常见题型：
    - 数组第k大的数
    - 链表的归并排序


通用交换函数：
```
void swap(int &a, int &b)
{
    int tmp = a;
    a = b;
    b = tmp;
}
```

* **插入排序**

第i个数前面序列已经排好序，将第i个数插入前面合适的位置

```
void insertSort(int a[], int n)
{
    for (int i = 1; i < n; i++)   // for i = 1:n-1
    {
        j = i;
        while (a[j] < a[j-1] && j > 0)
        {
            swap(a[j], a[j-1]);
            j--;
        }
    }
}
```

* **交换排序**(冒泡排序、快速快序)

冒泡排序：
```
void bubbleSort(int a[], int n)
{
    for (int i = 0; i < n-1; i++)    // for i = 0:n-2
    {
        for (int j = 0; j < n-1-i; j++)
        {
            if (a[j] > a[j+1]) swap(a[j], a[j+1]);
        }
    }
}
```

快速排序：
```
void quickSort(int a[], int left, int right)
{
    if (left < right)
    {
        int pivot = partition(a, left, right);
        quickSort(a, left, pivot);
        quickSort(a, pivot+1, right);
    }
}

int partition(int a[], int left, int right)
{
    int i = left;
    int j = right;
    while (i < j)
    {
        // 从右向左，找到第一个左边>右边的数
        while (i < j && a[i] <= a[j]) j--;
        // 满足i<j条件时，只能是左边的数大于右边的数，交换二者位置
        if (i < j) swap(a[i], a[j]);

        while (i < j && a[i] <= a[j]) i++;
        if (i < j) swap(a[i], a[j]);
    }
    return i;
}
```

归并排序：
```
L[right];    // 定义辅助存储空间，大小正比于元素个数
R[right];
void mergeSort(int a[], int left, int right)
{
    if (left < right)
    {
        int middle = (left + right)/2;
        mergeSort(a, left, middle);
        mergeSort(a, middle+1; right);
        merge(a, left, middle, right);
    }
}

void merge(int a[], int left, int middle, int right)
{
    int length1 = middle -left + 1;
    int length2 = right - middle;
    // 将两个子数组分别拷贝到L和R数组中
    for (int i = 0; i < length1; i++) L[i] = a[left + i];
    for (int i = 0; i < length2; i++) R[i] = a[middle + 1 + i];
    L[n1] = INT_MAX;
    R[n2] = INT_MAX;

    int i = 0;
    int j = 0;
    // 将L和R数组中的元素合并，并覆盖掉a中对应的元素
    for (int k = left; k <= right; k++)
    {
        if (L[i] < R[j]) 
            a[k] = L[i++];
        else 
            a[k] = R[j++];
    }
}
```

### 遍历

* **Binary tree traversal**

BFS, 即平常所说的层次遍历
```
vector<int> breadthFirstSearch(Node* root)
{
    vector<int> result;
    queue<Node*> nodeQueue;   // 使用队列，作为中间存储数据结构

    nodeQueue.push(root);
    Node* node;
    while(!nodeQueue.empty())
    {
        node = nodeQueue.front();   // 从队列中取出一个元素
        nodeQueue.pop();

        result.push_back(node->data);    // 将其元素加入最终vector中
        if (node->left) nodeQueue.push(node->left);        // 加入左节点
        if (node->right) nodeQueue.push(node->right);    // 加入右节点
    }
    return result;
}
```

ZigZag traversal
```
vector<vector<int> > zigzagLevelOrder(Node* root)
{
    vector<vector<int> result;   // 结果vector
    if (root==NULL) return result;

    vector<int> tmp;   // 临时vector
    flag = 0;
    queue<Node*> nodeQueue;   // 中间队列结构

    nodeQueue.push(root);    // 一次性push两个元素
    nodeQueue.push(NULL);

    while(!nodeQueue.empty())
    {
        node = nodeQueue.front();
        nodeQueue.pop();

        if (node!=NULL)   // 当取出的不为空时，就push下一层元素
        {
            tmp.push_back(node->data);
            if (node->left) nodeQueue.push(node->left);
            if (node->right) nodeQueue.push(node->right);
        }
        else    // 当取出的为空时，就push空值
        {
            if(!tmp.empty())
            {
                nodeQueue.push(NULL);

                // 判断是否调换顺序，Z字形存储
                if(flag == 1) reverse(tmp.begin(), tmp.end());
                flag = 1 - flag;
                result.push_back(tmp);
                tmp.clear();
            }
        }
    }
    return result;
}
```

* **DFS**

包括前序、中序、后序遍历

1. 先序：考察到一个节点后，即刻输出该节点的值，并继续遍历其左右子树。(根左右)
2. 中序：考察到一个节点后，将其暂存，遍历完左子树后，再输出该节点的值，然后遍历右子树。(左根右)
3. 后序：考察到一个节点后，将其暂存，遍历完左右子树后，再输出该节点的值。(左右根)

前序(递归)，其他递归遍历都类似，交换一下保存顺序即可
```
vector<int> result；
void preOrderTraversal(Node* root)
{
    if (root)
    {
        result.push_back(root->data);
        preOrderTraversal(result->left);
        preOrderTraversal(result->right);
    }
}
```

前序(非递归)，跟广度有个算法对应
```
vector<int> preOrderTraversal(Node* root)
{
    vector<int> result;
    stack<Node*> nodeStack;   // 使用栈，作为中间存储的数据结构

    nodeStack.push(root);
    Node* node;
    while(!nodeStack.empty())
    {
        node = nodeStack.pop();

        result.push_back(node->data);
        if (node->right) nodeStack.push(node->right);
        if (node->left) nodeStack.push(node->left);
    }
    return result；
}
public void preOrderTraverse2(TreeNode root) {  
    LinkedList<TreeNode> stack = new LinkedList<>();  
    TreeNode node = root;
    // 当遍历到最后一个节点的时候，无论它的左右子树都为空，并且栈也为空
    // 所以，只要不同时满足这两点，都需要进入循环
    while (node != null || !treeNodeStack.isEmpty()) {
        // 若当前考查节点非空，则输出该节点的值
        // 由考查顺序得知，需要一直往左走
        while (node != null) {
            System.out.print(node.val + " ");
            // 为了之后能找到该节点的右子树，暂存该节点
            treeNodeStack.push(node);
            node = node.left;
        }
        // 一直到左子树为空，则开始考虑右子树
        // 如果栈已空，就不需要再考虑
        // 弹出栈顶元素，将游标等于该节点的右子树
        if (!treeNodeStack.isEmpty()) {
            node = treeNodeStack.pop();
            node = node.right;
        }
    }
}
```

中序(递归)
```
void inOrderTraversal(Node* root)
{
    if (root)
    {
        InOrderTraversal(node->left);
        result.push_bach(node->data);
        InOrderTraversal(node->right);
    }
}
```

非递归中序遍历，跟前序很类似
```
public static void middleorderTraversal(TreeNode root) {
    Stack<TreeNode> treeNodeStack = new Stack<TreeNode>();
    TreeNode node = root;
    while (node != null || !treeNodeStack.isEmpty()) {
        while (node != null) {
            treeNodeStack.push(node);
            node = node.left;
        }
        if (!treeNodeStack.isEmpty()) {
            node = treeNodeStack.pop();
            System.out.print(node.val + " ");
            node = node.right;
        }
    }
}
```

后序(递归)
```
void postOrderTraversal(Node* node)
{
    if (root)
    {
        postOrderTraversal(node->left);
        postOrderTraversal(node->right);
        result.push_back(node->data);
    }
}
```

非递归后序遍历
```
public static void postorderTraversal(TreeNode root) {
    Stack<TreeNode> treeNodeStack = new Stack<TreeNode>();
    TreeNode node = root;
    // 输出当前节点的值的时候，需要考虑其左右子树是否都已经遍历完成
    // 若lastVisit等于当前考查节点的右子树，表示该节点的左右子树都已经遍历完成，则可以输出当前节点
    TreeNode lastVisit = root;
    while (node != null || !treeNodeStack.isEmpty()) {
        while (node != null) {
            treeNodeStack.push(node);
            node = node.left;
        }
        //查看当前栈顶元素
        node = treeNodeStack.peek();
        //如果其右子树也为空，或者右子树已经访问
        //则可以直接输出当前节点的值
        if (node.right == null || node.right == lastVisit) {
            System.out.print(node.val + " ");
            treeNodeStack.pop();
            lastVisit = node;
            node = null;
        } else {
            //否则，继续遍历右子树
            node = node.right;
        }
    }
}
```

### dynamic programming

* **minimal total in triangle**
```
int minimalTotal(vector<vector<int>> &triangle)
{
    for (int i = triangle.size() - 2; i >= 0; --i)
    {
        for (int j = 0; j < i + 1; ++j)
        {
            triangle[i][j] += max(triangle[i+1][j], triangle[i+1][j+1]);
        }
    }
    return triangle[0][0];
}
```

* **maximum subarray**

最大连续子序列和
```
int maxSubArray(int A[], int n)
{
    int result = INT_MIN;
    int f = 0;

    for (int i = 0; i < n; ++i)
    {
        f = max(f + A[i], f);
        result = max(result, f);
    }
    return result;
}
```

* **best time to buy and sell stock**

决定最佳购买股票时机

只能购买和出售一次，相当于寻找最大递增子串
```
int maxProfit(vector<int> &prices)
{
    if (prices.size() < 2) return 0;
    int profit = 0;
    int cur_min = prices[0];

    for (int i = 1; i < prices.size(); i++)
    {
        profit = max(profit, prices[i] - cur_min);
        cur_min = min(prices[i], cur_min);
    }
    return profit;
}
```

可以多次购买，即把所有为正的差价加起来
```
int maxProfit(vector<int> &prices)
{
    int profit = 0;
    for (int i = 1; i < prices.size(); i++)
    {
        profit += max(prices[i] - prices[i-1], 0);
    }
    return profit;
}
```


## 数据结构实现

### 堆实现

```
class MinHeap {
private:
    int capacity;
    int size;
    int* val;

private:
    void siftdown(int start, int end);
    void siftip(int start);

public:
    MinHeap(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        this.array = new int[capacity];
    }

    // 使某个坐标id的子堆变成最小堆
    void MinHeapify(int index) {
        int l = left(index);
        int r = right(index);
        int smallest = i;
        if (l < size && array[l] < array[smallest]) smallest = l;
        if (r < size && array[r] < array[smallest]) smallest = r;
        if (smallest != i) {
            swap(&array[smallest], &array[i]);
            MinHeapify(smallest);
        }
    }

    int parent(int i) {return (i-1)/2};
    int left(int i) {return 2*i+1};
    int right(int i) {return 2*+2};

    // 删除最小值，并使之成为新的最小堆
    int popMin() {
        if (size <= 0) {
            return INT_MAX;
        }
        if (size == 1) {
            size--;
            return array[0];
        }

        int minValue = array[0];
        size--;
        array[0] = array[size];
        MinHeapify(0);

        return minValue;
    }

    // 获取最小值
    int getMin() {return array[0];}


    // 插入元素，根据值
    void siftup(int val) {
        if (size == capacity) {
            cout << "Overflow of min heap";
            return;
        }
        size++;
        int index = size - 1;
        while (array[parent(index)] > array[index] && index != 0) {
            swap(&array[parent(index)], &array[index]);
            index = parent(index);
        }
    }
    // 删除元素，根据id
    void siftdown(int index) {
        array[i] = INT_MIN;
        while (array[parent(index)] > array[index] && index != 0) {
            swap(&array[parent(index)], &array[index]);
            index = parent(index);
        }
        popMin();

    }
};
```

