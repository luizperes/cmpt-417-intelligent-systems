#!/usr/bin/env python

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt

#--------------------------------------------------------------
#| Instance | Solution 1 | Solution 2   | Solution 2 w/o C10 |
#-------------------------------------------------------------
#| Test 1   | 149 msec   | 159 msec     | 152 msec           |
#| Test 2   | 153 msec   | 193 msec     | 154 msec           |
#| Test 3   | 176 msec   | 237 msec     | 218 msec           |
#| Test 4   | 863 msec   | 20s 419 msec | 17s 548msec        |
#| Test 5   | 170 msec   | 183 msec     | 190 msec           |
#| Test 6   | 144 msec   | 168 msec     | 128 msec           |
#| Test 7   | 171 msec   | 192 msec     | 169 msec           |
#| Test 8   | 138 msec   | 166 msec     | 164 msec           |
#| Test 9   | 155 msec   | 203 msec     | 155 msec           |
#| Test 10  | 162 msec   | 183 msec     | 164 msec           |
#| Test 11  | inf        | inf          | inf                |
#-------------------------------------------------------------
 
Test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Sol1 = [0.149, 0.153, 0.176, 0.863, 0.170, 0.144, 0.171, 0.138, 0.155, 0.162, float("inf")]
Sol2 = [0.159, 0.193, 0.237, 20.419, 0.183, 0.168, 0.192, 0.166, 0.203, 0.183, float("inf")]
Sol3 = [0.152, 0.154, 0.218, 17.548, 0.190, 0.128, 0.169, 0.164, 0.155, 0.164, float("inf")]

plt.figure()
plt.plot(Test, Sol1)
plt.plot(Test, Sol2)
plt.plot(Test, Sol3)
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.show()
