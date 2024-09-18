#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:02:47 2023

@author: dungdinh
"""
import random
# Given an integer array and a "target" number, 
# return indices of 2 numbers that they add to "target". 
# The input may contains more than one valid answer, you can return any of them.
# E.g: arr=[-1,2,3,4,1,-2], target = 0
# Answer: [0,4] or [1,5]

def sum_two_numbers(arr, target):
    # for i in range(0, len(arr)):
    #     for j in range(1, len(arr)):
    #         if arr[i] + arr[j] == target:
    #             return([i,j])
    
    find = []
    # for num in arr:
    #     find.append(target-num)
    
    for num in arr:
        find.append(target-num)
        if num in find:
            return [arr.index(num), find.index(num)]
        
        
# print(sum_two_numbers([-1,2,3,4,1,-2], 0))

#3 SUM
def sum_three_numbers(arr, target):
    i = 0
    j = i + 1
    
    arr_len = len(arr)
    
    two_num_list = []
    
    while i < arr_len-1:
        while j <arr_len:
            two_num_list.append([i, j, arr[i]+arr[j]])
            j = j+1
        i = i+1
        
            
    for index, value in enumerate(arr):
        for j in two_num_list:
            if target - j[2] == value:
                if (index != j[1]) and (index != j[0]): 
                    return [index, j[0], j[1]]

# print(sum_three_numbers([0, 0, 0, 5, 3], 0))  
# print(sum_three_numbers([1, -1, 2, 5, 3], 8))  

def sum_by_2_pointers(arr, target):
    arr.sort()
    #[-1, 1, 2, 3, 5]
    
    for index, value in enumerate(arr):
        s = index + 1
        e = len(arr) - 1
        
        while s < e:
            if (target - arr[s] - arr[e]) == 0:
                return [index, s, e]
            elif (target - arr[s] - arr[e]) > 0:
                s = s+1
            else:
                e = e -1
            
    

# print(sum_by_2_pointers([0, 0, 0, 5, 3], 0))
# print(sum_by_2_pointers([1, -1, 2, 5, 3], 8))

def isPalindrome(self, x):
    """
    :type x: int
    :rtype: bool
    """
    x_list = str(x)
    s = 0
    e = len(x_list)-1
    while s < e:
        if x_list[s] != x_list[e]:
            return False
        s = s+1
        e = e-1
    return True
        
# isPalindrome(121,121)

def merge(self, nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
    
    i = 0
    while i < n:
        nums1.pop()
        i = i + 1
    
    e2_step = 0
    
    while e2_step < n:
        print(e2_step)
        e1_step = 0
        while e1_step < m:
            if nums2[e2_step] <= nums1[e1_step]:
                nums1.insert(e1_step, nums2[e2_step])
                nums2.pop(e2_step)
                n -= 1
                m += 1
                e2_step -= 1
                break
            e1_step += 1
        e2_step += 1
    nums1.extend(nums2)
    print(nums1)
# merge(0, [4,0,0,0,0,0], 1, [1,2,3,5,6], 5)

def removeElement(self, nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    i = 0
    while i < len(nums):
        if nums[i] == val:
            nums.pop(i)
            i -= 1
        i += 1
        
    print(nums)
            
# removeElement(1, [0,1,2,2,3,0,4,2], 2)

def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    compare_list = []
    i=0
    while i < len(nums):
        if nums[i] in compare_list:
            nums.pop(i)
            i -= 1
        else:
            compare_list.append(nums[i])
        i += 1
    print(nums)
    
# removeDuplicates(1, [1,1,2])

def majorityElement(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    counts = {}
    for i in nums:
        if i in counts:
            counts[i] += 1
        else: counts[i] = 1
    return max(counts, key=counts.get)
# print(majorityElement(1, [2,2,1,1,1,2,2]))

def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    max_profit = 0
    start = 0
    end = 1
    while end < len(prices):
        if prices[start] < prices[end]:
            max_profit = max(max_profit, prices[end] - prices[start])
            end += 1
        else:
            start = end
            end += 1
    return max_profit
# print(maxProfit(1, [2,1,4]))

def lengthOfLastWord(self, s):
    """
    :type s: str
    :rtype: int
    """
    last_word = " ".join(s.split()).split(" ")[-1]
    return len(last_word)

# print(lengthOfLastWord(1, "   fly me   to   the moon  "))
def plusOne(self, digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    
    result = 0
    left = len(digits)-1
    while left >= 0:
        steps = len(digits)-left-1
        multiplier = 1
        while steps > 0:
            multiplier *= 10
            steps -= 1
        result = result + digits[left]*multiplier
        left -= 1
    
    return [int(x) for x in str(result+1)]
        
    
# print(plusOne(1, [1,2,3]))

def buyChoco(self, prices, money):
    """
    :type prices: List[int]
    :type money: int
    :rtype: int
    """
    prices.sort()
    if prices[0]+prices[1] <= money:
        return money - prices[0] - prices[1]
    else: return money
    
# print(buyChoco(1, [3,2,3], 3))
def maxWidthOfVerticalArea(self, points):
    """
    :type points: List[List[int]]
    :rtype: int
    """
    x_list = [i[0] for i in points]
    x_list.sort()
    length_pairs = []

    right = len(x_list)-1
    while right > 0:
        left = right - 1 
        length_pairs.append(x_list[right]-x_list[left])
        right -= 1
        
    return max(length_pairs)

# print(maxWidthOfVerticalArea(1, [[3,1],[9,0],[1,0],[1,4],[5,3],[8,8]]))

def lengthOfLIS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    
    buff_list = []
    
    i = 1
    last_append = 0
    while i < len(nums)-1:
        if nums[i] > nums[last_append]:
            if not buff_list:
                buff_list.append(nums[last_append])
                buff_list.append(nums[i])
                last_append = i
            else:
                buff_list.append(nums[i])
                last_append = i
        i += 1
    print(buff_list)
    return len(buff_list)
        
        
    
# print(lengthOfLIS(1, [0,1,0,3,2,3]))

def merge(self, nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
    
    nums1 = nums1[:-n]
    nums1 = nums1 + nums2
    nums1.sort()
    print(nums1)
    
# merge(1, [1,2,3,0,0,0], 3, [2,5,6], 3)

def romanToInt(self, s):
    """
    :type s: str
    :rtype: int
    """
    
    index = len(s)-1
    result = 0
    while index >= 0:
        if s[index] == 'I':
            result += 1
            index -= 1
        elif s[index] == 'V':
            if s[index-1] == 'I' and index-1 >= 0:
                result += 4
                index -= 2
            else:
                result += 5
                index -= 1
        elif s[index] == 'X':
            if s[index-1] == 'I' and index-1 >= 0:
                result += 9
                index -= 2
            else:
                result += 10
                index -= 1
        elif s[index] == 'L': 
            if s[index-1] == 'X' and index-1 >= 0:
                result += 40
                index -= 2
            else:
                result += 50
                index -= 1
        elif s[index] == 'C':
            if s[index-1] == 'X' and index-1 >= 0:
                result += 90
                index -= 2
            else:
                result += 100
                index -= 1
        elif s[index] == 'D':
            if s[index-1] == 'C' and index-1 >= 0:
                result += 400
                index -= 2
            else:
                result += 500
                index -= 1
        elif s[index] == 'M': 
            if s[index-1] == 'C' and index-1 >= 0:
                result += 900
                index -= 2
            else:
                result += 1000
                index -= 1
    return result
    
# print(romanToInt(1, "MMMCDXC"))

def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    
    index = 0
    count = 0
    while index <= len(nums)-2:
        if nums[index+1] == nums[index] and count == 0:
            count += 2
            index += 1
        elif nums[index+1] == nums[index] and count != 0:
            nums.pop(index+1)
        else:
            index += 1
            count = 0
    return(len(nums))
# print(removeDuplicates(1, [1,1,1,1]))

def rotate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    k=k%len(nums)
    nums[:] = nums[-k:] + nums[:-k]
    print(nums)
# rotate(1, [1, 2], 3)

def longestCommonPrefix(self, strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    max_letters = min([len(e) for e in strs])
    strs.sort()
    longest_common = ''
    
    check_index = 0
    while check_index < max_letters:
        if strs[0][check_index] == strs[-1][check_index]:
            longest_common = longest_common + strs[0][check_index]
        else:
            return longest_common
        check_index += 1
    return longest_common
    
  
# print(longestCommonPrefix(1, ["flower","flow","flight"]))

def strStr(self, haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    haystack_index = 0
    while haystack_index <= (len(haystack)-len(needle)):
        needle_index = 0
        while needle_index < len(needle) and (haystack_index + needle_index < len(haystack)):
            if (haystack[haystack_index + needle_index] == needle[needle_index]) and (needle_index == len(needle)-1):
                return haystack_index
            elif haystack[haystack_index + needle_index] == needle[needle_index]:
                needle_index += 1
            else: 
                needle_index = 0
                haystack_index += 1
    return -1
    
# print(strStr(1, "sadbutsad", "sad"))

def isPalindromee(self, s):
    """
    :type s: str
    :rtype: bool
    """
    s = s.lower()
    s = s.replace(" ", "")
    
    index = 0
    while index < len(s):
        if s[index].isalnum() == False:
            s = s[:index] + s[index+1:]
            index -= 1
        index += 1
    
    
    index1 = 0
    index2 = len(s)-1
    while index1 <= index2:
        if s[index1] != s[index2]:
            return False
        else:
            index1+=1
            index2-=1
    return True
    
# print(isPalindromee(1, "......a....."))

def isSubsequence(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    
    if s=="":
        return True
    
    t_index = 0
    while t_index < len(t):
        s_index = 0
        while s_index < len(s) and t_index < len(t): 
            if t[t_index] == s[s_index] and s_index == len(s)-1:
                return True
            elif t[t_index] == s[s_index]:
                s_index += 1
            elif t[t_index] != s[s_index] and t_index == len(t)-1:
                return False
            t_index += 1
        t_index += 1
    return False
# print(isSubsequence(1, "acb", "ahbgdc"))

def canConstruct(self, ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    def hash(str):
        letter_count = {}
        str_len = len(str)
        idx = 0
        while idx < str_len:
            if letter_count.get(str[idx]):
                letter_count.update({str[idx]: letter_count[str[idx]]+1})
            else:
                letter_count[str[idx]] = 1
            idx += 1
        return letter_count
    ransomNote_hashed = hash(ransomNote)
    magazine_hashed = hash(magazine)
    
    for ransomNote_key, ransomNote_value in ransomNote_hashed.items():
        if ransomNote_key not in magazine_hashed.keys():
            return False
        for magazine_key, magazine_value in magazine_hashed.items():
            if ransomNote_key == magazine_key and ransomNote_value > magazine_value:
                return False
    return True
                
      
# print(canConstruct(1, 'a', 'b'))

def canConstructt(self, ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    def hash(str):
        letter_count = {}
        str_len = len(str)
        idx = 0
        while idx < str_len:
            if letter_count.get(str[idx]):
                letter_count.update({str[idx]: letter_count[str[idx]]+1})
            else:
                letter_count[str[idx]] = 1
            idx += 1
        return letter_count
    magazine_hashed = hash(magazine)
    
    ransomNote_idx = 0
    while ransomNote_idx < len(ransomNote):
        if ransomNote[ransomNote_idx] in magazine_hashed.keys():
            if magazine_hashed[ransomNote[ransomNote_idx]] > 0:
                magazine_hashed.update({ransomNote[ransomNote_idx]: magazine_hashed[ransomNote[ransomNote_idx]]-1})
            else:
                return False
        else:
            return False
        ransomNote_idx += 1
    return True
               
      
# print(canConstructt(1, 'a', 'b'))
def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        def hash_character_with_positions(str):
            character_w_positions_dict = {}
            for idx, character in enumerate(str):
                if character_w_positions_dict.get(character) is not None:
                    character_w_positions_dict[character].append(idx)
                    character_w_positions_dict.update({character: character_w_positions_dict[character]})
                else:
                    character_w_positions_dict.update({character: [idx]})
            return character_w_positions_dict
        
        s_hashed = hash_character_with_positions(s)
        t_hashed = hash_character_with_positions(t)
        
        s_values = list(s_hashed.values())
        t_values = list(t_hashed.values())
        
        #compare 2 dictionary values
        s_idx = 0
        s_values_len = len(s_values)
        while s_idx < s_values_len:
            if s_values[s_idx] in t_values:
                if s_idx == s_values_len-1:
                    return True
                elif s_values[s_idx] in t_values:
                    t_values.remove(s_values[s_idx])
                    s_idx += 1
            else:
                return False
# print(isIsomorphic(1, "egg", "atd"))
def isIsomorphic_byMap_Find(s, t):
    return list(map(s.find, s)) == list(map(t.find, t))
    
# print(isIsomorphic_byMap_Find("egg", "add"))
def wordPattern(self, pattern, s):
    """
    :type pattern: str
    :type s: str
    :rtype: bool
    """
    s_splited = s.split(" ")
    def hash_character_with_positions(str):
        character_w_positions_dict = {}
        for idx, character in enumerate(str):
            if character_w_positions_dict.get(character) is not None:
                character_w_positions_dict[character].append(idx)
                character_w_positions_dict.update({character: character_w_positions_dict[character]})
            else:
                character_w_positions_dict.update({character: [idx]})
        return character_w_positions_dict
    pattern_hashed = list(hash_character_with_positions(pattern).values())
    str_hashed = list(hash_character_with_positions(s_splited).values())
    
    pattern_hashed.sort()
    str_hashed.sort()
    return pattern_hashed == str_hashed
    
# print(wordPattern(1, "abba", "dog cat cat fish"))

def test_word_pattern(pattern, str):
    s = pattern
    t = str.split()
    print(list(map(s.find, s)))
    print(list(map(t.index, t)))
    return list(map(s.find, s)) == list(map(t.index, t))
# print(test_word_pattern("abba", "dog cat cat fish"))

def isAnagram(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    
    def counter(str):
        dict = {}
        for e in str:
            if e in dict:
                dict.update({e: dict[e]+1})
            else:
                dict.update({e: 1})
        return dict
    return counter(s) == counter(t)
        
# print(isAnagram(1, "anagram", "nagaram"))
def isHappy(self, n):
    """
    :type n: int
    :rtype: bool
    """
    while len(str(n)) >= 1:
        n_elements = str(n)
        n_squared = [int(e)*int(e) for e in n_elements]
        if sum(n_squared) == 1:
            return True
        else: n = sum(n_squared)
# print(isHappy(1, 19))

def containsNearbyDuplicate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    location_hash = {}
    for index, value in enumerate(nums):
        if value in location_hash:
            location_hash.update({value: location_hash[value] + [index]})
        else:
            location_hash[value] = [index]
    
    for value in location_hash:
        locations = location_hash[value]
        
        i = 0
        location_len = len(locations)
        while i < location_len - 1:
            next_idx = i + 1
            if locations[next_idx]-locations[i] <= k:
                return True
            else:
                i += 1
    return False
# print(containsNearbyDuplicate(1, [1,2,3,1,2,3], 2))

def summaryRanges(self, nums):
    """
    :type nums: List[int]
    :rtype: List[str]
    """
    res = []
    for i in nums:
        if res and res[-1][1] == i-1:
            res[-1][1] = i
        else:
            res.append([i,i])
    return [str(x)+"->"+str(y) if x!=y else str(x) for x,y in res]
            
            
    # print(full_nums)
# print(summaryRanges(1, [0,1,2,4,5,7]))

def isValid(self, s):
    """
    :type s: str
    :rtype: bool
    """
    dict = {"[": "]",
            "{": "}",
            "(": ")"}
    stack = []
    for c in s:
        if c in "[{(":
            stack.append(dict[c])
        elif c in "]})" and (stack == [] or stack.pop() != c):
            return False

    return stack == []
        
    
# print(isValid(1, "]"))

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
head = ListNode(3)
head.next = ListNode(2)
head.next.next = ListNode(0)
head.next.next.next = ListNode(-4)
head.next.next.next.next = head.next

def hasCycle(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    if head == []:
        return False
    
    while head.next:
        if head.next.val != 'visited':
            head.next.val = 'visited'
            head = head.next
        else:
            return True
    return False
        

# print(hasCycle(1, []))

#list1 creation
list1 = ListNode(1)
list1.next = ListNode(2)
list1.next.next = ListNode(4)

#list 2 creation
list2 = ListNode(1)
list2.next = ListNode(3)
list2.next.next = ListNode(4)

def mergeTwoLists(self, l1, l2):
    """
    :type list1: Optional[ListNode]
    :type list2: Optional[ListNode]
    :rtype: Optional[ListNode]
    """
    dump = cur = ListNode(0)
    
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dump.next
    
        
# print(mergeTwoLists(1, list1, list2))

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
node1 = TreeNode(9)
node2 = TreeNode(15)
node3 = TreeNode(7)
node4 = TreeNode(20)
node5 = TreeNode(3)
node5.left = node1
node5.right = node4
node4.left = node2
node4.right = node3

def maxDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def dfs(root, depth):
        if root == None:
            return depth
        else:
            return max(dfs(root.left, depth + 1), dfs(root.right, depth + 1))
    return dfs(root, 0)
        
    
    
# print(maxDepth(1, node5))

nodep_1 = TreeNode(1)
nodep_2 = TreeNode(2)
nodep_3 = TreeNode(3)

nodep_1.left = nodep_2
nodep_1.right = nodep_3

nodeq_1 = TreeNode(1)
nodeq_2 = TreeNode(2)
nodeq_2.left = TreeNode(3)
nodeq_2.right = TreeNode(4)
nodeq_3 = TreeNode(2)
nodeq_3.left = TreeNode(4)
nodeq_3.right = TreeNode(3)

nodeq_1.left = nodeq_2
nodeq_1.right = nodeq_3

def isSameTree(self, p, q):
    """
    :type p: TreeNode
    :type q: TreeNode
    :rtype: bool
    """
    if p == None and q == None:
        return True
    elif p == None or q == None:
        return False
    elif p.val == q.val:
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    else:
        return False
    
def invertTree(self, root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    if root == None:
        return
    else:
       root.right, root.left = root.left, root.right
       invertTree(1, root.left)
       invertTree(1, root.right)
       return root
# print(invertTree(1, nodeq_1))

def isSymmetric(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    def check_symmetric(left, right):
        if not left and not right:
            return True
        elif not left or not right:
            return False
        else:
            return left.val == right.val and check_symmetric(left.left, right.right) and check_symmetric(left.right, right.left)
    
    if not root:
        return True
    return check_symmetric(root.left, root.right)
# print(isSymmetric(1, nodeq_1))

def hasPathSum(self, root, targetSum):
    """
    :type root: TreeNode
    :type targetSum: int
    :rtype: bool
    """
    print(root)
    if not root:
        return False
    
    if not root.left and not root.right and root.val == targetSum:
        return True
    
    targetSum -= root.val
    return hasPathSum(1, root.left, targetSum) or hasPathSum(1, root.right, targetSum)
    
# print(hasPathSum(1, nodeq_1, 6))

def in_order_traverse(root):
    def recursion(root, data):
        if not root:
            return []
        else:
            if root.left:
                data += recursion(root.left, [])
                
            data.append(root.val)
            
            if root.left:
                data += recursion(root.right, [])
                
            return data
    return recursion(root, [])
# print(in_order_traverse(nodeq_1))

def post_order_traverse(root):
    
    def recursion(root,data):
        if not root:
            return 0
        else:       
            if root.left:
                data += recursion(root.left, 0)
            
            if root.right:
                data += recursion(root.right, 0)
                
            data += 1
            
            return data
    return recursion(root, 0)
# print(post_order_traverse(nodeq_1))

def averageOfLevels(self, root):
    """
    :type root: TreeNode
    :rtype: List[float]
    """
    if not root:
        return 0

    res = []
    current_level = [root]
    
    while current_level:
        next_level = []
        val_list = []
        for node in current_level:
            val_list.append(node.val)
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        res.append(sum(val_list)/len(val_list))
        current_level = next_level
    return res
# print(averageOfLevels(1, nodeq_1))

def getMinimumDifference(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def get_val_list(root):
        if not root:
            return 0
    
        res = []
        current_level = [root]
        
        while current_level:
            next_level = []
            for node in current_level:
                res.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            current_level = next_level
        return res
    
    val_list = get_val_list(root)
    min_absolute = max(val_list)
    start1 = 0
    while start1 < len(val_list)-1:
        start2 = start1+1
        while start2 < len(val_list):
            if abs(val_list[start2] - val_list[start1]) < min_absolute:
                min_absolute = abs(val_list[start2] - val_list[start1])
            start2 += 1
        start1 += 1
    return min_absolute
    # val_list = sorted(val_list)
    # print(val_list[-1]-val_list[-2])
# print(getMinimumDifference(1, nodeq_1))


def sortedArrayToBST(self, nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """ 
    def recursion(nums):
        if not nums:
            return None
        
        middle = len(nums)//2
        
        root_node = TreeNode(nums[middle])
        root_node.left = recursion(nums[:middle])
        root_node.right = recursion(nums[middle+1:])
        
        return root_node

    root = recursion(nums)
    return root
# sortedArrayToBST(1, [-10, -3, 0, 5, 9])

def searchInsert(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    # middle = len(nums)//2
    # if target == nums[middle]:
    #     return middle
    # elif target > nums[middle]:
    #     right_list = nums[middle+1:]
    #     right_middle = len(right_list)//2
        
    #     if target <= right_list[right_middle]:
    #         return middle + right_middle + 1
    #     elif target > right_list[right_middle]:
    #         return middle + right_middle + 2
    # else:
    #     left_list = nums[:middle]
    #     left_middle = len(left_list)//2
        
    #     if target <= left_list[left_middle]:
    #         return left_middle 
    #     elif target > left_list[left_middle]:
    #         return left_middle + 1
    # def recursion(nums, start):
    #     middle = len(nums)//2
    #     if nums[middle] == target:
    #         return start - middle
        
    #     if middle == 0:
    #         if target > nums[middle]:
    #             return middle + start + 1
    #         else:
    #             if start - middle < 0:
    #                 return 0
    #             return start - middle
    #     else:
    #         if target > nums[middle]:
    #             if not nums[middle+1:]:
    #                 return middle + start + 1
    #             start = start + middle + 1
    #             return recursion(nums[middle+1:], start)
    #         else:
    #             start = middle - 1
    #             return recursion(nums[0:middle], start)
  
    # return recursion(nums, 0)
    for index, val in enumerate(nums):
        if val >= target:
            return index
    return len(nums)
    
# print(searchInsert(1, [1,2,3,4,5,10], 2))

def addBinary(self, a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    len_a = len(a)
    len_b = len(b)
    
    num_0 = abs(len(a)-len(b))
    add = 1
    change_num = b if len_a > len_b else a
    while add <= num_0:
        change_num = "0" + change_num
        add += 1
    if len_a > len_b:
        b = change_num
    else:
        a= change_num
    
    max_len = len_a if len_a > len_b else len_b
    
    index = max_len-1
    result = ""
    redundant = 0
    while index >= 0:  
            if int(a[index]) == int(b[index]) == 1:
                if redundant == 0:
                    result = "0" + result
                else:
                    result = "1" + result
                redundant = 1
            elif int(a[index]) == int(b[index]) == 0:
                if redundant == 0:
                    result = "0" + result
                else:
                    result = "1" + result
                redundant = 0
            else:
                if redundant == 0:
                    result = "1" + result
                    redundant = 0
                else:
                    result = "0" + result
                    redundant = 1
            index -= 1
    return "1" + result if redundant == 1 else result
        
# print(addBinary(1, "1010", "1011"))

def reverseBits(self, n):
    # def squared_by(num):
    #     if num == 0:
    #         return 1
    #     else:
    #         result = 1
    #         count = 1
    #         while count <= num:
    #             result = result*2
    #             count += 1
    #         return result
    
    # n_str = str(n)
    # index = 0
    # result = 0
    # while index < len(n_str):
    #     if n_str[index] == "1":
    #         result += squared_by(index)
    #     index += 1
    # return result
    out = 0
    for i in range(32):
        out = (out << 1)^(n&1)
        n >>= 1
    return out
        
            
    
# print(reverseBits(1, 11111111111111111111111111111101))

def hammingWeight(self, n):
    """
    :type n: int
    :rtype: int
    """
    def max_square_check(num):
        if num == 1:
            return 0
        else:
            max = 1
            while max <= num:
                max *= 2
            return num - max/2
    
    count = 0
    remain = n
    while remain != 0:
        remain = max_square_check(remain)
        count += 1
    return count
            
# print(hammingWeight(1, 128))

def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    index = len(nums)-1
    while index >= 0:
        if len(nums) == 1:
            return nums[0]
        else:
            val = nums[index]
            nums.remove(val)
            if val not in nums:
                return val
            else:
                nums.remove(val)
                index -= 2
    
    
# print(singleNumber(1, [1, 0, 1]))

def mySqrt(self, x):
    """
    :type x: int
    :rtype: int
    """
    l, r = 0, x

    while l <= r:
        mid = l + (r-l)//2
        if mid*mid <= x < (mid+1)*(mid+1):
            return mid
        elif mid*mid > x:
            r = mid - 1
        else:
            l = mid + 1
    
# print(mySqrt(1, 1))

def maxProfit2(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    index = 0
    prices_len = len(prices)
    buy = prices[0]
    profits = []
    while index < prices_len-1:
        if prices[index+1] >= prices[index]:
            sell = prices[index+1]
            if index == prices_len - 2 and sell:
                profits.append(sell-buy)
                break
        elif prices[index+1] < prices[index]:
            #start selling
            sell = prices[index]
            profits.append(sell-buy)
            buy = prices[index+1]
        index+=1
    print(profits)
    
    return sum(profits)
    
# print(maxProfit2(1, [1,9,6,9,1,7,1,1,5,9,9,9]))

def fibonaci(memory, n):
    if n in memory:
        return memory[n]
    
    if n == 0 or n == 1:
        return 1
    
    memory.update({n: fibonaci(memory, n-1) + fibonaci(memory, n-2)})
    return memory[n]
# print(fibonaci(5))

def run():
    memory = {}
    fibonaci(memory, 5)
    print(memory)
# run()

def fibonaci_memory(step, memory, n):
    if step < n:
        memory.append(fibonaci(step-1)+fibonaci(step-2))
    
    if n == 0 or n == 1:
        return 1
    return fibonaci(n-1) + fibonaci(n-2)

def canJump(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    if len(nums) == 1:
        return True
    
    index = len(nums)-2
    steps_needed = 1
    while index >= 0:
        if index == 0 and nums[index] >= steps_needed:
            return True
        elif nums[index] >= steps_needed:
            steps_needed = 1
        else:
            steps_needed += 1
        index -= 1
    return False
    
    
# print(canJump(1, [0]))

def jump(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if len(nums) == 1:
        return 0
    
    def recursion(no_of_jumps, nums):
        if len(nums) == 0:
            return no_of_jumps
        else:
            steps = range(len(nums)-1, -1, -1)
            index = 0
            while index <= len(nums)-1:
                if nums[index] >= steps[index]:
                    no_of_jumps += 1
                    if len(nums[:index]) > 1:
                        index += 1
                    nums = nums[:index]
                    
                    return recursion(no_of_jumps, nums)
                index += 1
    return recursion(0, nums)
    
    
# print(jump(1, [3, 2, 1]))
def jump_reinstall(nums):
    l = r = 0
    nJumps = 0
    while r < len(nums)-1:
        furthest = max(i + nums[i] for i in range(l, r+1))
        nJumps += 1
        l, r = r+1, furthest
    return nJumps
        
    
# print(jump_reinstall([1, 1, 1, 1]))

def hIndex(self, citations):
    """
    :type citations: List[int]
    :rtype: int
    """
    citations.sort(reverse=True)
    return sum(v > i for i, v in enumerate(citations))
    
# print(hIndex(1, [1, 1, 1, 1]))

def intToRoman(self, num):
    """
    :type num: int
    :rtype: str
    """
    def num_to_elements(num):
        element = 1
        result = []
        num_len = len(str(num))
        index = 1
        
        while index <= num_len:
            result.insert(0, num%(element*10))
            num = num - num%(element*10)
            element *= 10
            index += 1
        return result
    elements = num_to_elements(num)
    print(elements)
    result = ''
    for e in elements:
        if e < 4000 and e > 1000:
            e = e//1000
            while e > 0:
                result = result + ('M')
                e -= 1
        if e == 1000:
            result = result + ('M')
        elif e == 900:
            result = result + ('CM')
        elif e < 900 and e > 500:
            result = result + 'D'
            e -= 500
            e = e // 100
            while e > 0:
                result = result + ('C')
                e -= 1
        elif e == 500:
            result = result +  ('D')
        elif e == 400:
            result = result +  ('CD')
        elif e < 400 and e > 100:
            e = e//100
            while e > 0:
                result = result + ('C')
                e -= 1
        elif e == 100:
            result = result +  ('C')
        elif e == 90:
            result = result +  ('XC')
        elif e < 90 and e > 50:
            result = result + 'L'
            e = e//10-5
            while e > 0:
                result = result + ('X')
                e -= 1
        elif e == 50:
            result = result +  ('L')
        elif e == 40:
            result = result +  ('XL')
        elif e < 40 and e > 10:
            e = e//10
            while e > 0:
                result = result + ('X')
                e -= 1
        elif e == 10:
            result= result +  ('X')
        elif e == 9:
            result = result + ('IX')
        elif e < 9 and e > 5:
            result = result + 'V'
            e -= 5
            while e > 0:
                result = result + ('I')
                e -= 1
        elif e == 5:
            result = result +  ('V')
        elif e == 4:
            result = result +  ('IV')
        elif e < 4 and e > 1:
            while e > 0:
                result = result + ('I')
                e -= 1
        elif e == 1:
            result = result + ('I')
        else:
            result 
    return result
            

# print(intToRoman(1, 2000))

def canCompleteCircuit(self, gas, cost):
    """
    :type gas: List[int]
    :type cost: List[int]
    :rtype: int
    """
    # if gas == cost:
    #     return 0
    
    # possible_indexes = []
    # index = 0
    # while index < len(gas):
    #     if gas[index] >= cost[index]:
    #         possible_indexes.append(index)
    #     index += 1
    
    #scan possible indexes
    index = 0
    while index < len(gas):
        count = 1
        start_index = index
        gas_fill = gas[start_index]
        while count <= len(gas):
            if count == len(gas) and gas_fill - cost[start_index] >= 0:
                return index
            elif gas_fill - cost[start_index] >= 0:
                next_index = 0 if start_index == len(gas) - 1 else start_index + 1
                gas_fill = gas_fill - cost[start_index] + gas[next_index]
                start_index = 0 if start_index == len(gas)-1 else start_index + 1
                count += 1
            else:
                break
        index += 1
    return -1
        

# print(canCompleteCircuit(1, [0, 0, 0, 0], [0, 0, 0, 0]))

def reverseWords(self, s):
    """
    :type s: str
    :rtype: str
    """
    #Remove spaces at beginning and ending
    s = s.strip()
    
    #Remove not alphabet after a space between words
    elements = s.split(' ')
    # print(elements)
    
    result = []
    for e in elements:
        if e.isalnum():
            result.append(e)
    result.reverse()
    
    # print(result)
    
    return " ".join(e for e in result)
    
# print(reverseWords(1, "EPY2giL"))

def twoSum(self, numbers, target):
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    index = 0
    need_numbers = {}
    while index < len(numbers):
        if numbers[index] in need_numbers:
            return [need_numbers[numbers[index]] + 1, index+1]
        else:
            need_numbers.update({target-numbers[index]: index})
        index += 1
    
# print(twoSum(1, [-1,0], -1))

def threeSum(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    
    #calculate 2sum
    two_sum_list = []
    index = 0
    while index < len(nums)-1:
        index2 = index + 1
        while index2 < len(nums):
            two_sum_list.append([index, index2, nums[index] + nums[index2]])
            index2 += 1
        index += 1
        
    #find another one
    result = []
    for idx, e in enumerate(nums):
        index = 0
        while index < len(two_sum_list):
            if ((two_sum_list[index][2] + e == 0) and (two_sum_list[index][0] != two_sum_list[index][1]) 
            and (two_sum_list[index][0]!= idx) and (two_sum_list[index][1]!= idx)):
                result.append(sorted([nums[idx], nums[two_sum_list[index][0]], nums[two_sum_list[index][1]]]))
            index += 1
            
    return_results = []
    for e in result:
        if e not in return_results:
            return_results.append(e)
    return return_results
        
# print(threeSum(1, [-1,0,1,2,-1,-4]))

def three_sum(self, nums):
    result = set()
    
    #create 3 lists:
    n, p, z = [], [], []
    for num in nums:
        if num < 0:
            n.append(num)
        elif num > 0:
            p.append(num)
        else:
            z.append(num)
    
    #2 sets of positives and negatives
    P, N = set(p), set(n)
    
    #At least 1 zero add -num + num:
    if z:
        for negative in N:
            if -1*negative in P:
                result.add((negative, 0, -1*negative))
    
    if len(z) >= 3:
        result.add((0, 0, 0))
    
    #check all pairs of negatives + 1 positive:
    for i in range(len(n)):
        for j in range(i+1, len(n)):
            if -1*(n[i] + n[j]) in P:
                result.add(tuple(sorted([n[i], n[j], -1*(n[i] + n[j])])))
                
    #check all pairs of positives + 1 negative:
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            if -1*(p[i] + p[j]) in N:
                result.add(tuple(sorted([-1*(p[i] + p[j]), p[i], p[j]])))
    return result
    
# print(three_sum(1, [1,1,-2]))

def maxArea(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    max_water = 0
    for i in range(len(height)):
        for j in range(i+1, len(height)):
            if abs(i-j)*min(height[j], height[i]) > max_water:
                max_water = abs(i-j)*min(height[j], height[i])
    return max_water
# print(maxArea(1, [1,1]))

def max_area(self, height):
    water_max = 0
    l, r = 0, len(height)-1
    while l < r:
        water_max = max(water_max, (r-l)*min(height[l], height[r]))
        if height[l] > height[r]:
            r -= 1
        else:
            l+= 1
    return water_max
# print(max_area(1, [1, 1]))

def productExceptSelf(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    # index = 0 
    # result = []
    # while index < len(nums):
    #     elements = nums[:index] + nums[index+1:]
    #     product = 1
    #     for e in elements:
    #         product *= e
    #     result.append(product)
    #     index += 1
    # return result
    products = []
    nums_len = len(nums)
    for i in range(nums_len):
        product = 1
        for j in range(len(nums)):
            if i == j:
                continue
            else:
                product *= nums[j]
        products.append(product)
    return products
# print(productExceptSelf(1, [1, 2, 3]))

def productExceptSelf1(self, nums):
    n = len(nums)
    answser = [1]*n
    
    #left pass:
    left=1
    for i in range(n):
        answser[i] = left
        left *= nums[i]
    
    #right pass:
    right = 1
    for i in range(n-1, -1, -1):
        answser[i] *= right
        right *= nums[i]
    
    print(answser)
    
# productExceptSelf1(1, [1, 2, 3])    

def isValidSudoku(self, board):
    """
    :type board: List[List[str]]
    :rtype: bool
    """
    #Checking rows
    for row in board:
        availables = []
        for e in row:
            if e == ".":
                continue
            elif e not in availables:
                availables.append(e)
            else:
                return False
    
    #Checking columns
    column = 0
    while column < 9:
        row = 0
        row_nums = []
        while row < 9:
            if board[row][column] != '.' and  board[row][column] not in row_nums:
                row_nums.append(board[row][column])                
            elif board[row][column] in row_nums:
                return False
            row += 1
        column += 1
        
    #Checking subboxes
    for i in (0, 3, 6):
        for j in (0, 3, 6):
            square = [board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            availables = []
            for num in square:
                if num != '.' and  num not in availables:
                    availables.append(num)                
                elif num in availables:
                    return False
    return True
# print(isValidSudoku(1, [["8","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]]))

def minSubArrayLen(self, target, nums):
    """
    :type target: int
    :type nums: List[int]
    :rtype: int
    """
    left=total=0
    result = len(nums)+1
    for right, num in enumerate(nums):
        total += num
        while total >= target:
            result = min(result, right - left + 1)
            total -= nums[left]
            left += 1
    return result if result <= len(nums) else 0
        
# print(minSubArrayLen(1, 213, [1, 1, 1]))

class RandomizedSet(object):

    def __init__(self):
        self.nums, self.pos = [], {}
        

    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if val in self.nums:
            return False
        else:
            self.nums.append(val)
            self.pos[val] = len(self.nums)
            return True
        

    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if val not in self.nums:
            return False
        else:
            idx, last = self.pos[val], self.nums[-1]
            self.nums[idx], self.pos[last] = last, idx
            self.nums.pop()
            self.pos.pop(val)
            return True
            
        

    def getRandom(self):
        """
        :rtype: int
        """
        return self.nums[random.randint(0, len(self.nums)-1)]
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(1)
# param_1 = obj.insert(12)
# # print(param_1)
# param_1 = obj.remove(1)
# print(param_1)
# param_3 = obj.getRandom()

def lengthOfLongestSubstring(self, s):
    """
    :type s: str
    :rtype: int
    """
    
    result = 0
    for i in range(len(s)):
        index = i
        temp = []
        while index < len(s):
            if s[index] not in temp and index == len(s) -1:
                temp.append(s[index])
                result = max(result, len(temp))
                break
            elif s[index] not in temp:
                temp.append(s[index])
                index += 1
            else:
                result = max(result, len(temp))
                break
    return result
            
    
# print(lengthOfLongestSubstring(1, " "))

def length_of_longest_substring(self, s):
    max_length = 0
    left = 0
    char_set = set()
    for right in range(len(s)):
        if s[right] not in char_set:
            char_set.add(s[right])
            max_length = max(max_length, right-left+1)
        else:
            while s[right] in char_set:
                char_set.remove(char_set[left])
                left += 1
            char_set.add(s[right])
    return max_length
        
    
# print(length_of_longest_substring(1, " "))

def rotate1(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: None Do not return anything, modify matrix in-place instead.
    """
    matrix[:] = zip(*matrix[::-1])
    print(matrix)
    
# rotate1(1, [[1,2,3],[4,5,6],[7,8,9]])

def setZeroes(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: None Do not return anything, modify matrix in-place instead.
    """
    positions = []
    for row in range(len(matrix)):
        for column in range(len(matrix[0])):
            if matrix[row][column] == 0:
                positions.append([row, column])
                # for i in range(len(matrix)):
                #     matrix[i][column] = 0
                # for i in range(len(matrix[0])):
                #     matrix[row][i] = 0
    
    for position in positions:
        row = position[0]
        column = position[1]
        for i in range(len(matrix)):
            matrix[i][column] = 0
        for i in range(len(matrix[0])):
            matrix[row][i] = 0
            
    return matrix
    
# print(setZeroes(1, [[0,1,2,0],[3,4,5,2],[1,3,1,5]]))

def longestConsecutive(self, nums) -> int:
    nums = list(set(nums))
    nums.sort()
    
    if len(nums) == 1:
        return 1
    
    
    index = 0
    nums_len = len(nums)
    result = 0
    count = 1
    while index < nums_len-1:
        if nums[index+1] - nums[index] == 1 and index == nums_len - 2:
            return max(result, count + 1)
        elif nums[index+1] - nums[index] == 1:
            count += 1
        else:
            result = max(result, count)
            count = 1
        index += 1
    return result
        
 
# print(longestConsecutive(1, [100,4,200,1,3,2]))
from collections import Counter
def groupAnagrams(self, strs):
    """
    :type strs: List[str]
    :rtype: List[List[str]]
    """
    counter_list = [Counter(str) for str in strs]
    seen_list = []
    result = []
    
    for i in range(len(counter_list)):
        equal_set = []
        for j in range(i+1, len(counter_list)):
            if counter_list[i] == counter_list[j] and counter_list[j] not in seen_list and counter_list[i] not in seen_list:
                equal_set.append(counter_list[j])
                
        if len(equal_set) == 0:
            result.append(counter_list[i])
        else:
            # print(equal_set)
            result.append([counter_list[i]] + equal_set)     
            seen_list.append(counter_list[i])
            seen_list.extend(equal_set)
    return result
        
                
# print(groupAnagrams(1, ["eat","tea","tan","ate","nat","bat"]))

def group_anagrams(self, strs):
    hash_dict = {}
    for string in strs:
        str_sorted = "".join(sorted(string))
        if str_sorted not in hash_dict:
            hash_dict.update({str_sorted: [string]})
        else:
            hash_dict[str_sorted].append(string)
    return hash_dict.values()
# print(group_anagrams(1, ["eat","tea","tan","ate","nat","bat"]))

def merge_overlapped(self, intervals):
    """
    :type intervals: List[List[int]]
    :rtype: List[List[int]]
    """
    intervals.sort()
    
    overlapped = intervals[0]
    result = []
    for i in range(1, len(intervals)):
        if intervals[i][0] <= overlapped[-1]:
            if intervals[i][0] <  overlapped[0]:
                first = intervals[i][0]
            else:
                first =  overlapped[0]
            
            if intervals[i][-1] < overlapped[-1]:
                last = overlapped[-1]
            else:
                last = intervals[i][-1]
            overlapped = [first, last]
        else:
            result.append(overlapped)
            overlapped = intervals[i]
    result.append(overlapped)
    return result
            
            

# print(merge_overlapped(1, [[1,4],[4,5]]))

def insert(self, intervals, newInterval):
    """
    :type intervals: List[List[int]]
    :type newInterval: List[int]
    :rtype: List[List[int]]
    """
    if not intervals and newInterval:
        return [newInterval]
    
    result = []
    overlapped = newInterval
    for i in range(len(intervals)):
        compare_list = newInterval if not overlapped else overlapped
        if intervals[i][1] >= compare_list[0] and intervals[i][0] <= compare_list[1]:
            #merge
            first = min(intervals[i][0], compare_list[0])
            last = max(intervals[i][1], compare_list[1])
            overlapped = [first, last]
        elif overlapped == newInterval:
            result.append(intervals[i])
        else:
            result.append(overlapped)
            overlapped = intervals[i]
    result.append(overlapped)
    result.sort()
    return result
    
# print(insert(1, [[1,5]], [0,0]))

def findMinArrowShots(self, points):
    """
    :type points: List[List[int]]
    :rtype: int
    """
    points.sort()
    print(points)
    merged = points[0]
    count = 0
    for i in range(1, len(points)):
        if points[i][0] <= merged[1]:
            first = min(merged[0], points[i][0])
            last = min(merged[1], points[i][1])
            merged = [first, last]
        else:
            count += 1
            merged = points[i]
    return count + 1 if merged else count
# print(findMinArrowShots(1, [[9,12],[1,10],[4,11],[8,12],[3,9],[6,9],[6,7]]))
import copy

def gameOfLife(self, board):
    """
    :type board: List[List[int]]
    :rtype: None Do not return anything, modify board in-place instead.
    """
    row_len = len(board)
    print('row_len', row_len)
    column_len = len(board[0])
    print('column_len', column_len)
    
    def marking(r, c, board_rep):
        resident_n = 0
        #left, right
        if c-1 >= 0 and board_rep[r][c-1] == 1:
            print('left')
            resident_n += 1
        if c+1 <= column_len -1 and board_rep[r][c+1] == 1:
            print('right')
            resident_n += 1
        #upper, lower
        if r-1 >= 0 and board_rep[r-1][c] == 1:
            print('upper')
            resident_n += 1
        if r+1 <= row_len-1 and board_rep[r+1][c] == 1:
            print('lower')
            resident_n +=1
        #diagonal
        if r-1 >= 0 and c-1 >= 0 and board_rep[r-1][c-1] == 1:
            print('upper left')
            resident_n += 1
        if r-1 >= 0 and c+1 <= column_len-1 and board_rep[r-1][c+1] == 1:
            print('upper right')
            resident_n += 1
        if r+1 <= row_len-1 and c-1 >= 0 and board_rep[r+1][c-1] == 1:
            print('lower left')
            resident_n += 1
        if r+1 <= row_len-1 and c+1 <= column_len-1 and board_rep[r+1][c+1] == 1:
            print('lower right')
            resident_n += 1
            
        # if(r == 1 and c == 0):
        #     print(c-1 >= 0)
        #     print('left: ', board_rep[r][c-1])
        #     print('right: ', board_rep[r][c+1])
        #     print('upper: ',  board_rep[r-1][c])
        #     print('lower: ', board_rep[r+1][c])
            
        #     print('upper left: ', board_rep[r-1][c-1])
        #     print('upper right: ', board_rep[r-1][c+1])
        #     print('lower left: ', board_rep[r+1][c-1])
        #     print('lower right: ', board_rep[r+1][c+1])
            
        if board_rep[r][c] == 1:
            if resident_n < 2:
                board[r][c] = 0
            elif resident_n <= 3:
                board[r][c] = 1 
            else:
                board[r][c] = 0
        else:
            if resident_n == 3:
                board[r][c] = 1
                
    board_rep = copy.deepcopy(board)
    for i in range(row_len):
        for j in range(column_len):
            print('i', i)
            print('j', j)
            
            if i-1 >= 0 and j+1 <= column_len-1:
                print('board_rep[0][1]', board_rep[0][1])
                print('upper righttt: ', board_rep[i-1][j+1])
            print('............')
            marking(i, j, board_rep)
    return board
            
    
# print(gameOfLife(1, [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]))

def simplifyPath(self, path):
    """
    :type path: str
    :rtype: str
    """
    parts = path.split('/')
    
    
    stack = []
    for part in parts:
        if stack and part == '..':
            stack.pop()
        elif part == '' or part == '/' or part == '.' or part == '..':
            continue
        else:
            stack.append(part)
    return '/' + '/'.join(stack)
        


# print(simplifyPath(1, "/../"))

def evalRPN(self, tokens):
    """
    :type tokens: List[str]
    :rtype: int
    """
    stack = []
    for token in tokens:
        if token not in ['+', '-', '*', '/']:
            stack.append(token)
        else:
            number2 = int(stack.pop())
            number1 = int(stack.pop())
            
            if token == '+':
                stack.append(number1+number2)
            elif token == '-':
                stack.append(number1-number2)
            elif token == '*':
                stack.append(number1*number2)
            else:
                stack.append(float(number1)/number2)
        
    return int(stack[0])
# print(evalRPN(1, ["4","-2","/","2","-3","-","-"]))  

add1 = ListNode(1)
add1.next = ListNode(2)
add1.next.next = ListNode(3)
add1.next.next.next = ListNode(4)
add1.next.next.next.next = ListNode(5)

add2 = ListNode(5)
add2.next = ListNode(6)
add2.next.next = ListNode(4)

def addTwoNumbers(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    dummy = ListNode(0)
    tail = dummy
    carry = 0
    while l1 is not None or l2 is not None or carry != 0:
        val1 = l1.val if l1 != None else 0
        val2 = l2.val if l2 != None else 0
        
        sum_val = val1 + val2 + carry
        
        digit = sum_val%10
        carry = sum_val//10
        
        newNode = ListNode(digit)
        tail.next = newNode
        tail = tail.next
        
        l1 = l1.next if l1 is not None else None
        l2 = l2.next if l2 is not None else None
        
    result = dummy.next
    dummy = None
    return result
    
# print(addTwoNumbers(1, add1, add2))

def reverseBetween(self, head, left, right):
    """
    :type head: ListNode
    :type left: int
    :type right: int
    :rtype: ListNode
    """
    dummy = ListNode(0)
    dummy.next = head
    
    pre = ListNode(0)
    pre = dummy
    cur = ListNode(0)
    cur = pre.next
    
    for _ in range(left-1):
        pre = pre.next
        cur = pre.next
        
    temp = None
    for _ in range(right - left):
        temp = cur.next
        cur.next = temp.next
        temp.next = pre.next
        pre.next = temp
        
    return dummy.next
# print(reverseBetween(1, add1, 2, 4))

def removeNthFromEnd(self, head, n):
    """
    :type head: ListNode
    :type n: int
    :rtype: ListNode
    """
    #get vals 
    vals = []
    while head is not None:
        vals.append(head.val)
        head = head.next
        
    del_pos = len(vals)-n
    vals.pop(del_pos)
    
    result = dummy = ListNode(0)
    for e in vals:
        result.next = ListNode(e)
        result = result.next

    return dummy.next
    
# removeNthFromEnd(1, add1, 2)

duplicate = ListNode(1)
duplicate.next = ListNode(2)
duplicate.next.next = ListNode(3)

def deleteDuplicates(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    dummy = pre = ListNode(0)
    dummy.next = head
    
    while head and head.next:
        if head.val == head.next.val:
            while head and head.next and head.val == head.next.val:
                head = head.next
            head = head.next
            pre.next = head
        else:
            pre = pre.next
            head = head.next
    return dummy.next
# print(deleteDuplicates(1, duplicate))

test = TreeNode(1)
test.left = TreeNode(2)
test.right = TreeNode(3)

def sumNumbers(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def dfs(root, cur):
        if not root:
            return 0
        cur = cur*10 + root.val
        if not root.right and not root.left:
            return cur
        return dfs(root.left, cur) + dfs(root.right, cur)
        
    return dfs(root, 0)
            
    
# print(sumNumbers(1, test))
def climbStairs(self, n):
    """
    :type n: int
    :rtype: int
    """
    
    def recursion(n, dic):
        if n in dic:
            return dic[n]
        
        if n <= 1:
            return 1
        
        dic.update({n: recursion(n-1, dic)+recursion(n-2, dic)})

        return dic[n]
    return recursion(n, {})
    
# print(climbStairs(1, 3))

def canFinish(self, numCourses, prerequisites):
    """
    :type numCourses: int
    :type prerequisites: List[List[int]]
    :rtype: bool
    """
    pre = {}
    prerequisites.sort()
    for course in prerequisites:
        if course[0] == course[1]:
            return False
        elif course[1] in pre and pre[course[1]] == course[0]:
            return False
        elif course[0] not in pre:
            if course[0] not in pre.values():
                pre.update({course[0]: course[1]})
            else:
                for e in pre:
                    if pre[e] == course[0]:
                        pre.update({e: course[1]})
    return True
    
# print(canFinish(1, 2, [[1,0],[0,2],[2,1]]))

def generateParenthesis(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    def generate(p, left, right, parens=[]):
        if left:         generate(p + '(', left-1, right)
        if right > left: generate(p + ')', left, right-1)
        if not right:    
            parens += p,
        return parens
    return generate('', n, n)
# print(generateParenthesis('', 3))

def letterCombinations(self, digits):
    """
    :type digits: str
    :rtype: List[str]
    """
    mapping = {
        '0': '',
        '1': '',
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
        }
    
    all_combinations = [''] if len(digits) > 0 else []
    
    for digit in digits:
        current_combination = []
        for letter in mapping[digit]:
            for combination in all_combinations:
                current_combination.append(combination+letter)
        all_combinations = current_combination
    return all_combinations

# print(letterCombinations(1, '23'))

test = ListNode(1)
test.next = ListNode(2)


def swapPairs(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return None
    
    vals = []
    while head:
        vals.append(head.val)
        head = head.next
    
    index = 0
    vals_len = len(vals)
    result = []
    while index < vals_len:
        if index < vals_len-1:
            result.append(vals[index+1])
            result.append(vals[index])
            index += 2
        else:
            result.append(vals[index])
            index += 1
            
    dummy = res = ListNode(result[0])
    for e in result[1:]:
        res.next = ListNode(e)
        res = res.next
    return dummy
# print(swapPairs(1, []))

duplicate1 = ListNode(1)
duplicate1.next = ListNode(1)
duplicate1.next.next = ListNode(2)
duplicate1.next.next.next = ListNode(3)

def deleteDuplicates2(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    dummy = pre = ListNode(0)
    dummy.next = head
    
    seen = []
    while head:
        if head.val in seen and head.next:
            pre.next = head.next
            head = head.next
        elif head.val in seen:
            pre.next = None
            head = None
        else:
            seen.append(head.val)
            pre.next = head
            pre = pre.next
            head = head.next
    return dummy.next
            
            
    
# print(deleteDuplicates2(1, duplicate1))

def reverseList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    pre = cur = None
    while head:
        cur = head
        head = head.next
        cur.next = pre
        pre = cur
        
    return pre
root = TreeNode(1)
node2 = TreeNode(2)
node3 = TreeNode(3)
node4 = TreeNode(4)
node5 = TreeNode(5)
node6 = TreeNode(6)
node7 = TreeNode(7)
root.left = node2
root.right = node3
node2.left = node4
node2.right = node5
node3.left = node6
node3.right = node7
def inorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    return inorderTraversal(self, root.left) + [root.val] + inorderTraversal(self, root.right) if root else []  
# print(inorderTraversal(1, root))

def generate(self, numRows):
    """
    :type numRows: int
    :rtype: List[List[int]]
    """
    def row_create(prev_list):
        if len(prev_list) == 0:
            return [1]
        
        res = []
        res.append(1)
        index = 0
        while index < len(prev_list)-1:
            res.append(prev_list[index]+prev_list[index+1])
            index += 1
        res.append(1)
        return res
    
    count = 0
    pre = []
    result = []
    while count <= numRows:
        cur = row_create(pre)
        result.append(cur)
        pre = cur
        count += 1
    return result[-1]
# print(generate(1, 0))

def traverse_inorder_wo_recursion(root):
    cur = root
    stack = []
    result = []
    while True:
        if cur is not None:
            stack.append(cur)
            cur = cur.left
        elif stack:
            cur = stack.pop()
            result.append(cur.val)
            cur = cur.right
        else:
            return result
# print(traverse_inorder_wo_recursion(root))

def traverse_postorder(root):
    return traverse_postorder(root.left) + traverse_postorder(root.right) + [root.val] if root else []
# print(traverse_postorder(root))

testin = ListNode(4)
testin.next = ListNode(1)
testin.next.next = ListNode(8)
testin.next.next.next = ListNode(4)
testin.next.next.next.next = ListNode(5)

test1 = ListNode(5)
test1.next = ListNode(6)
test1.next.next = ListNode(1)
test1.next.next.next = ListNode(8)
test1.next.next.next.next = ListNode(4)
test1.next.next.next.next.next = ListNode(5)

def getIntersectionNode(self, headA, headB, intersectVal):
    """
    :type head1, head1: ListNode
    :rtype: ListNode
    """
    def get_vals(head):
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res
    headA_vals = get_vals(headA)
    headB_vals = get_vals(headB)

    A_index = len(headA_vals)-1
    B_index = len(headB_vals)-1
    intersect = []
    while A_index and B_index:
        if headA_vals[A_index] == headB_vals[B_index]:
            intersect.append(headA_vals[A_index])
            A_index -= 1
            B_index -= 1
        else:
            break
    intersect.reverse()
            
    if not intersect:
        return None
    else:
        dummy = res = ListNode(intersect[0])
        for e in intersect[1:]:
            res.next = ListNode(e)
            res = res.next
        return dummy
    
# print(getIntersectionNode(1, testin, test1, 8))

def removeElements(self, head, val):
    """
    :type head: ListNode
    :type val: int
    :rtype: ListNode
    """
    dummy = first = ListNode(0)
    dummy.next = head
    
    while head:
        if head.val == val:
           head = head.next
        else:
           first.next = head
           head = head.next
           first = first.next
    first.next = None
    return dummy.next.next.next.next.val
            
# print(removeElements(1, testin, 5)) 
def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """ 
    if target in nums:
        return nums.index(target)
    else:
        return -1
    
# print(search(1, [4,5,6,7,0,1,2], 0))

def rotateRight(self, head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """
    if not head:
        return None
    
    #lenght of head 
    vals = []
    while head:
        vals.append(head.val)
        head = head.next 
    
    k=k%len(vals)
    vals[:] = vals[-k:] + vals[:-k]

    dummy = res = ListNode(vals[0])
    for e in vals[1:]:
        res.next = ListNode(e)
        res = res.next
    res.next = None
    return dummy
# print(rotateRight(1, duplicate, 1))

def minDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return 0
    return 1 + (min(minDepth(1, root.left), minDepth(1, root.right)) or max(minDepth(1, root.left), minDepth(1, root.right)))
    
# print(minDepth(1, root))
def containsDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    
    def hash(nums):
        count_vals = {}
        for e in nums:
            if e not in count_vals:
                count_vals.update({e: 1})
            else:
                old_val = count_vals[e]
                count_vals.update({e: old_val+1})
        return count_vals
    
    nums_dic = hash(nums)
    for e in nums_dic:
        if nums_dic[e] > 1:
            return True
    return False
    
# print(containsDuplicate(1, [1,2,3,4]))

head = ListNode(1)
head.next = ListNode(4)
head.next.next = ListNode(3)
head.next.next.next = ListNode(0)
head.next.next.next.next = ListNode(2)
head.next.next.next.next.next = ListNode(5)
head.next.next.next.next.next.next = ListNode(2)

def partition(self, head, x):
    """
    :type head: ListNode
    :type x: int
    :rtype: ListNode
    """
    if not head:
        return None
    #get vals
    vals = []
    while head:
        vals.append(head.val)
        head = head.next
    
    #partition by x 
    x_idx = vals.index(x)
    before_x = vals[:x_idx] if vals[:x_idx] else []
    after_x = vals[x_idx+1:] if vals[x_idx+1:] else []
    
    #insert
    def insert_before_x(beforeX, val):
        if len(beforeX) == 0 and val:
            return [val]
        
        beforeX_len = len(beforeX)
        idx = beforeX_len-1
        while idx >= 0:
            if val <= beforeX[idx]:
                return beforeX[:idx] + [val] + beforeX[idx:]
            else:
                idx-=1
        return beforeX + [val]
    #rearrange
    idx = 0
    after_x_len = len(after_x)
    while idx < after_x_len:
        if after_x[idx] < x:
            before_x = insert_before_x(before_x, after_x[idx])
        idx += 1

    after_x = [ele for ele in after_x if ele > x]
    
    #merge parts


    res = before_x + [x] + after_x
    
    #return ListNode
    dummy = start = ListNode(res[0])
    for val in res[1:]:
        start.next = ListNode(val)
        start = start.next
    return dummy
# print(partition(1, head, 3))  
def sortList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return None
    vals = []
    while head:
        vals.append(head.val)
        head = head.next
        
    vals.sort()
    dummy = start = ListNode(vals[0])
    for val in vals[1:]:
        start.next = ListNode(val)
        start = start.next
    return dummy
# sortList(1, head)

def trailingZeroes(self, n):
    """
    :type n: int
    :rtype: int
    """
    def factorize(n):
        res = 1
        while n >= 1:
            res = res*n
            n -= 1
        return res
    
    n_factorize = str(factorize(n))
    
    count=0
    idx = len(n_factorize)-1
    
    while idx >= 0:
        if n_factorize[idx] == '0':
            count += 1
            idx -= 1
        else:
            return count
            
# print(trailingZeroes(1, 7))

def permute(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    resutl = []
    #base case
    if len(nums)==1:
        return [nums[:]]
    
    for i in range(len(nums)):
        n = nums.pop(0)
        perms = permute(1, nums)
        
        for perm in perms:
            perm.append(n)
        resutl.extend(perms)
        nums.append(n)
    return resutl
        
        
# print(permute(1, [1, 2, 3]))

def binaryTreePaths(self, root):
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    if not root:
        return []
    
    res = []
    def dfs(node, path):
        if not node:
            return
        
        path += str(node.val)
        
        if not node.left and not node.right:
            res.append(path)
            return
        
        if node.left:
            dfs(node.left, path + "->")
        if node.right:
            dfs(node.right, path + "->")
    dfs(root, "")
    return res
    
# print(binaryTreePaths(1, root))

def addDigits(self, num):
    """
    :type num: int
    :rtype: int
    """
    if len(str(num)) == 1:
        return num
    
    else:
        s = 0
        for n in str(num):
            s+= int(n)
        num = s
        return addDigits(1, num)
    
# print(addDigits(1, 38))

def addStrings(self, num1, num2):
    """
    :type num1: str
    :type num2: str
    :rtype: str
    """
    num1_index = len(num1)-1
    num2_index = len(num2)-1
    mem = 0
    res = ''
    while num1_index >= 0 or num2_index >= 0:
        if num1_index >= 0 and num2_index >= 0:
            sum = int(num1[num1_index]) + int(num2[num2_index]) + mem
            res = str(sum%10) + res
            mem = int(sum/10)
            num1_index -=1
            num2_index -= 1
        elif num1_index < 0:
            sum = int(num2[num2_index]) + mem
            res = str(sum%10) + res
            mem = int(sum/10)
            num2_index -= 1
        else:
            sum = int(num1[num1_index]) + mem
            res = str(sum%10) + res
            mem = int(sum/10)
            num1_index -= 1
    return str(mem) + res if mem > 0 else res
# print(addStrings(1, '1', '9'))

def countAndSay(self, n):
    """
    :type n: int
    :rtype: str
    """
    def say(count):
        if count == '0':
            return '1'
        len_count = len(count)
        idx = 0
        res = ''
        c = 1
        while idx < len_count:
            if idx + 1 >= len(count) or count[idx+1] != count[idx]:
                res = res + str(c) + count[idx]
                c = 1
            else:
                c += 1
            idx += 1
        return res
    
    idx = 0
    res = '0'
    while idx < n:
        res = say(res)
        print(idx)
        print(res)
        idx += 1
    return res
# print(countAndSay(1, 6))

def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    check_arr = range(n+1)
    for num in check_arr:
        if num not in nums:
            return num
    
# print(missingNumber(1, [1]))

def moveZeroes(self, nums):
    """
    :type nums: List[int]
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    index = len(nums)-1
    count = 0
    while index >=0:
        if nums[index]==0:
            nums.pop(index)
            count += 1
        index -= 1

    while index < count-1:
        nums += [0]
        index += 1
     
# moveZeroes(1, [0,1,0,3,12])

def thirdMax(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums_set = set(nums)
    num_list = list(nums_set)
    num_list.sort(reverse=True)
    return num_list[2] if len(num_list) >= 3 else max(num_list)
    
# print(thirdMax(1, [2,2,3,1]))

def countSegments(self, s):
    """
    :type s: str
    :rtype: int
    """

    s=s.strip()
    if not s:
        return 0
    
    s=s.split(' ')
    index = len(s)-1
    count = 0
    while index >= 0:
        if s[index] != '':
            count += 1
        index -= 1
    return count
            
            
# print(countSegments(1, "Hello"))

def arrangeCoins(self, n):
    """
    :type n: int
    :rtype: int
    """
    res = 0
    while n >= res + 1:
        n -= res+1
        res +=1 
    return res
    
# print(arrangeCoins(1, 1))

def findDisappearedNumbers(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    s = set(nums)
    return [i for i in range(1, len(nums)+1) if i not in s]
# print(findDisappearedNumbers(1, [1, 1]))

def combine(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    result = []
    def backtrack(start, combination):
        if len(combination) == k:
            result.append(combination[:])
            return
        for i in range(start, n+1):
            print('i', i)
            print('combination', combination)
            combination.append(i)
            backtrack(i+1, combination)
            combination.pop()
    backtrack(1, [])
    return result
# combine(1, 4, 2)

def combinationSum(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    def backtrack(start, target, path):
        if target == 0:
            result.append(path)
            return
        if target < 0:
            return  # Pruning: No need to continue if the target is negative
        for i in range(start, len(candidates)):
            # Choose the candidate and explore further
            backtrack(i, target - candidates[i], path + [candidates[i]])

    result = []
    backtrack(0, target, [])
    return result
    
# print(combinationSum(1, [2,3,6,7], 7))

def exist(self, board, word):
    """
    :type board: List[List[str]]
    :type word: str
    :rtype: bool
    """
    def backtrack(i, j, k):
        if k == len(word):
            return True
        
        if i<0 or i>=len(board) or j<0 or j >= len(board[0]) or board[i][j] != word[k]:
            return False
        
        tmp = board[i][j]
        board[i][j] = ''
        
        if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
            return True
        
        board[i][j] = tmp
        return False
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            if backtrack(i, j, 0):
                return True
    return False
        
    
# exist(1, [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED")
    
def isHappy1(self, n):
    """
    :type n: int
    :rtype: bool
    """
    def next_n(n):
        return sum([int(digit)**2 for digit in str(n)])
        
    seen = set()
    while n!= 1 and n not in seen:
        seen.add(n)
        n = next_n(n)
    return n==1
# print(isHappy1(1, 19))


def merge1(self, nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: None Do not return anything, modify nums1 in-place instead.
    """
    k = m+n-1
    m = m-1
    n = n-1
    #merge while 2 lists have num
    while m >= 0 and n >= 0:
        if nums1[m] >= nums2[n]:
            nums1[k] = nums1[m]
            m -= 1
            
        else:
            nums1[k] = nums2[n]
            n -= 1
        k-=1 
    
    #remaining nums
    while n >= 0:
        nums1[k] = nums2[n]
        n -= 1
        k -= 1
    return nums1
    
        
# print(merge1(1, [1,2,3,0,0,0], 3, [2,5,6], 3))   

def removeElement1(self, nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    n = len(nums)
    for i in range(n-1, -1, -1):
        if nums[i] == val:
            nums.pop(i)
        i -= 1
    return len(nums)
# print(removeElement1(1, [3,2,2,3], 3))

def removeDuplicates1(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    cur_num = None
    count = 0
    nums_len = len(nums)-1
    for i in range(nums_len, -1, -1):
        if nums[i] != cur_num:
            cur_num = nums[i]
            count = 1
        else:
            if count <= 1:
                count += 1
            else:
                nums.pop(i)
        i-=1 
    return len(nums)
# print(removeDuplicates1(1, [0,0,1,1,1,1, 1,2,2,3,3,4]))

def candy(self, ratings):
    """
    :type ratings: List[int]
    :rtype: int
    """
    res = [0]*len(ratings)
    for i, val in enumerate(ratings[:-1]):
        if val >= ratings[i+1]:
            res[i] = 2
            res[i+1] = 1
        else:
            res[i] = 1
            res[i+1] = 2
    print(res)
# candy(1, [1,3,2,2,1])

def convert(self, s, numRows):
    """
    :type s: str
    :type numRows: int
    :rtype: str
    """
    if numRows==1 or len(s) == 1:
        return s
    #add to rows
    res = ['' for _ in range(numRows)]
    
    down =False
    row = 0
    for c in s:
        res[row] += c
        if row == 0 or row == numRows-1:
            down = not down
        if down:
            row += 1
        else: row -= 1
        
    #merge rows
    return ''.join(res)
# print(convert(1,  "AB", 1))

from collections import deque

def maxDepthBFS(root: TreeNode) -> int:
    if root is None:
        return 0
    queue = deque([root])
    depth = 0
    
    
    while queue:
        level_size = len(queue)
        print(level_size)
        for _ in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        depth += 1
    
    return depth
# print(maxDepthBFS(root))

def isValidSudoku1(self, board):
    #checking rows

    def checkValid(row):
        exist = set()
        for val in row:
            if val in exist and val != '.':
                return False
            else:
                exist.add(val)
        return True
    
    for row in board:
        if not checkValid(row):
            return False
                
    #checking columns
    for column in range(9):
        column_vals = []
        for row in board:
            column_vals.append(row[column])
        if not checkValid(column_vals):
            return False
    
    #checking sub-boxes
    for row in range(0, 9, 3):
        for column in range(0, 9, 3):
            sub_box_vals = []
            for r in range(row, row+3):
                for c in range(column, column+3):
                    sub_box_vals.append(board[r][c])
            if not checkValid(sub_box_vals): return False
    return True
    

# print(isValidSudoku1(1, [["5","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]]))

def isValid1(self, s: str):
    #corresponding open parentheses
    pairs = {
        ")": "(",
        "}": "{",
        "]": "["
        }
    seen_open = []
    close_parentheses = ")}]"
    for parentheses in s:
        if parentheses not in close_parentheses:
            seen_open.append(parentheses)
        else:
            if not seen_open:
                return False
            nearest = seen_open.pop()
            if pairs[parentheses] != nearest:
                return False
            
    return len(seen_open)==0
        
    
# print(isValid1(1, "()[]("))

def generateParenthesis1(self, n: int):
    res = []
    def backtrack(cur, open_count, close_count):
        if len(cur) == 2*n:
            res.append(cur)
            return
        
        if open_count < n:
            backtrack(cur + '(', open_count+1, close_count)
            
        if close_count < open_count:
            backtrack(cur + ')', open_count, close_count + 1)
    
    backtrack("", 0, 0)
    return res
        
# print(generateParenthesis1(1, 3))
def combine1(n, k):
    res = []
    def backtrack(start, path):
        
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(start, n+1):
            print(path)
            path.append(i)
            backtrack(i+1, path)
            path.pop()
    backtrack(1, []) 
    return res
        
# print(combine1(3, 2))

def permute1(nums):
    def backtrack(path, used):
        print("call backtrack", path, used)
        if len(path) == len(nums):
            res.append(path[:])
            return
        

        
        # path = [1], used=[True, False, False]
        for i in range(len(nums)):
            if not used[i]:
                # path [1,3]
                path.append(nums[i])
                # [True, False, True]
                used[i] = True
                backtrack(path, used)
            
                # path [1,2]
                path.pop()
                # path [1]
                print("pop path=", path)
                
                # [True, False, False]
                used[i] = False
                print('used=', used)
    res = []
    used = [False]*len(nums)
    backtrack([], used)
    return res 
# print(permute1([1,2,3]))

def permute2(nums):
    n = len(nums)
    def backtrack(path, remain):
        if len(path) == n:
            print('PATH', path)
            res.append(path[:])
            return
        
        for i in range(len(remain)):
            print('remain=', remain)
            print('path=', path)
            path.append(remain[i])
            backtrack(path, remain[:i] + remain[i+1:])
            path.pop()
    res=[]
    backtrack([], nums)
    return res
# print(permute2([1,2,3]))
import heapq
def findKthLargest(self, nums, k):
    # while k>1:
    #     max_num = max(nums)
    #     nums.remove(max_num)
    #     k -= 1
    # return max(nums)
    heap = heapq.nlargest(k, nums)
    
    # Return the k-th largest element (which is the smallest element in the heap)
    return heap[-1]

print(findKthLargest(1, [3,2,3,1,2,4,5,5,6], 4)) 
