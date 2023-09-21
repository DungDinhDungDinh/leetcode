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

def returnIndicesofTarget(arr, target):
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
        
        
print(returnIndicesofTarget([-1,2,3,4,1,-2], 0))