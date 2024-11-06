from collections import deque, defaultdict


#1. Two Pointers
#Useful for problems involving arrays or linked lists,
# particularly when searching pairs, finding palindromes, and more.



def tow_sum(arr, target):
    left = 0
    right = len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]


print(tow_sum([1, 2, 3, 4, 5], 9))


#2. Sliding Window
#Great for substring and contiguous subarray problems with fixed or dynamic window sizes.

def lenght_of_longest_substring(s):
    char_set = set()
    left =  max_length = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    return max_length


print(lenght_of_longest_substring("abcabcbb"))


#3. Binary Search
#Efficient for sorted arrays and searching for elements or ranges.

def binary_search(arr, target):
    left = 0 
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
        
    return -1


print(binary_search([1, 2, 3, 4, 5], 3))


#4. Fast & Slow Pointers
#Useful for detecting cycles in linked lists and finding the middle element.

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next             # Move slow pointer by 1 step
        fast = fast.next.next        # Move fast pointer by 2 steps
        if slow == fast:
            return True              # Cycle detected
    return False 


print(has_cycle(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))))


#5. Depth-First Search (DFS)
#Used in tree and graph traversal, backtracking, and pathfinding.

def max_area_of_island(grid):
    def dfs(i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != 1:
            return 0
        grid[i][j] = 0
        return 1 + dfs(i - 1, j) + dfs(i + 1, j) + dfs(i, j - 1) + dfs(i, j + 1)
    max_area = 0
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                max_area = max(max_area, dfs(i, j))
    return max_area

print(max_area_of_island([[0,0,1,0,0,0,0,1,0,0,0,0,0],
                           [0,0,0,0,0,0,0,1,1,1,0,0,0],
                           [0,1,1,0,1,0,0,0,0,0,0,0,0],
                           [0,1,0,0,1,1,0,0,1,0,1,0,0],
                           [0,1,0,0,1,1,0,0,1,1,1,0,0],
                           [0,0,0,0,0,0,0,0,0,0,1,0,0],
                           [0,0,0,0,0,0,0,1,1,1,0,0,0],
                           [0,0,0,0,0,0,0,1,1,0,0,0,0]]))

#6. Breadth-First Search (BFS)
#Useful for shortest path problems in unweighted graphs.

def shotest_path_binary_matrix(grid):
    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1
    queue = deque([(0, 0)])
    grid[0][0] = 1
    while queue:
        row, col = queue.popleft()
        if row == len(grid) - 1 and col == len(grid[0]) - 1:
            return grid[row][col]
        for r, c in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0:
                queue.append((r, c))
                grid[r][c] = grid[row][col] + 1
    return -1



print(shotest_path_binary_matrix([[0,0,0],
                                  [1,1,0],
                                  [1,1,0]]))


#7. Merge Intervals
#Often used in problems requiring overlapping intervals to be merged.


def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged



print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))


#8. Topological Sort
#Ideal for dependency resolution problems in directed acyclic graphs (DAGs).

def can_finish(num_courses, prerequisites):
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    visited = set()
    def dfs(course):
        if course in visited:
            return False
        visited.add(course)
        for prereq in graph[course]:
            if not dfs(prereq):
                return False
        return True
    for course in range(num_courses):
        if not dfs(course):
            return False
    return True

print(can_finish(2, [[1,0]]))


#9. Dynamic Programming (DP) - Bottom-Up
#Used for problems with overlapping subproblems, like sequence problems or optimization problems.

def climb_stairs(n):
    if n <= 2:
        return n 
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

print(climb_stairs(10))

#10. Union-Find (Disjoint Set)
#Useful for connected component problems and cycle detection in undirected graphs.

def count_components(n, edges):
    perent = list(range(n))
    
    def find(x):
        if x != perent[x]:
            perent[x] = find(perent[x])
        return perent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            perent[px] = py
            
            
            
    for x, y in edges:
        union(x, y)
        
    return len(set(find(x) for x in range(n)))


print(count_components(5, [[0, 1], [1, 2], [3, 4]]))

#2. DP with 2D Grid - Unique Paths
#This pattern is often used for grid traversal problems
# where you need to find the number of ways to get from one cell to another with specific movement constraints.

def unnique_paths(m, n):
    dp = [[0] * n for _ in range(m)]
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
    return dp[m-1][n-1]

print(unnique_paths(3, 7))


    
    
#3. DP for Subset Sum - Knapsack Pattern
#This pattern is used for problems involving subset selection under constraints.
# The classic example is the "0/1 Knapsack" problem.

def can_partition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    return dp[target]


print(can_partition([1, 2, 3, 9]))
 







    
    


       
    
    


        
       
