#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:02:47 2023

@author: dungdinh
"""
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
    
  
print(longestCommonPrefix(1, ["flower","flow","flight"]))