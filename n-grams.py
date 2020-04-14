from nltk import ngrams
import math
import numpy as np
from random import randrange
from tqdm import tqdm, trange
import time

# rolling hash
def rolling_hash(Base, bucket_size, gram,n):

    hash = 0
    for i in range(n):
        hash += (gram[i])*(Base**(n-1-i))
    last_gram = gram
    yield hash % bucket_size

    while True:
        following_gram = yield
        hash = hash * Base - (last_gram[0])*Base**n + (following_gram[n-1])
        last_gram = following_gram
        yield hash % bucket_size


def partition(x, pivot_index = 0):
    i = 0
    if pivot_index !=0: x[0],x[pivot_index] = x[pivot_index],x[0]
    for j in range(len(x)-1):
        if x[j+1] > x[0]:
            x[j+1],x[i+1] = x[i+1],x[j+1]
            i += 1
    x[0],x[i] = x[i],x[0]
    return x,i

def RSelect(x,k):
    if len(x) == 1:
        return x[0]
    else:
        xpart = partition(x,randrange(len(x)))
        x = xpart[0] # partitioned array
        j = xpart[1] # pivot index
        if j == k:
            return x[j]
        elif j > k:
            return RSelect(x[:j],k)
        else:
            k = k - j - 1
            return RSelect(x[(j+1):], k)
        
# Python program to find top k elements in a stream 
# Function to print top k numbers 
def kTop(a, n, k): 
    k = k
    # list of size k+1 to store elements 
    top = [0 for i in range(k + 1)] 

    # dictionary to keep track of frequency 
    freq = {i:0 for i in range(k + 1)} 

    # iterate till the end of stream 
    for m in range(n): 

        # increase the frequency 
        if a[m] in freq.keys(): 
            freq[a[m]] += 1
        else: 
            freq[a[m]] = 1

        # store that element in top vector 
        top[k] = a[m] 

        i = top.index(a[m]) 
        i -= 1
        
        while i >= 0: 

            # compare the frequency and swap if higher 
            # frequency element is stored next to it 
            if (freq[top[i]] < freq[top[i + 1]]): 
                t = top[i] 
                top[i] = top[i + 1] 
                top[i + 1] = t 
            
            # if frequency is same compare the elements 
            # and swap if next element is high 
            elif ((freq[top[i]] == freq[top[i + 1]]) and (top[i] > top[i + 1])): 
                t = top[i] 
                top[i] = top[i + 1] 
                top[i + 1] = t 
            else: 
                break
            i -= 1
        
    # return top k elements 
    return top[:-1]


def main():

    n = 12
    base = 3
    bucket_size = 2**31 -19
    s = math.ceil(n/4)
    k = 24
    global bucket, T 
    T = []
    sample_list = ['brbbot.exe','brbbot.exe']
    print("Generating N-grams Hash")
    for sample in tqdm(sample_list):
        with open(sample, 'rb') as f:
            bytes_array = np.array(bytearray(f.read()), dtype="uint8")
        f.close()
        grams = ngrams(bytes_array, n)
        count = 0
        for gram in grams:
            if count == 0:
                gen = rolling_hash(base, bucket_size, gram, n)
                key = next(gen)
            else:
                next(gen)
                key = gen.send(gram)
            
            if key % s == 0:
                T.append(key)
            count += 1
        # T.append(grams)
    Top_K = kTop(T, len(T), k) 
    # print(len(Top_K))
    print(Top_K)
    print("Restoring N-grams")
    for sample in tqdm(sample_list):
        with open(sample, 'rb') as f:
            bytes_array = np.array(bytearray(f.read()), dtype="uint8")


        grams = ngrams(bytes_array, n)
        count = 0
        used_key = []
        for gram in grams:
            if count == 0:
                gen = rolling_hash(base, bucket_size, gram, n)
                key = next(gen)
            else:
                next(gen)
                key = gen.send(gram)

            if key in Top_K and key not in used_key:
                used_key.append(key)
                if count == 0:
                    bucket = np.array([list(gram)])
                else:
                    bucket = np.append(bucket, [list(gram)], axis=0)
                count += 1
    print(bucket)
    # Kth_item = RSelect(T, k-1)
    # print(Kth_item)




if __name__ == "__main__":
    main()
    # T = [5,6,12,2.5,46,7,8,1,2,3,5,10,10,10,10,1,1,2,2]
    # print(kTop(T, len(T), 2))