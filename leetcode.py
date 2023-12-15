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
print(majorityElement(1, [2,2,1,1,1,2,2]))
