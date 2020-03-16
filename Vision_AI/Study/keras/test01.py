# print("동달이")

# import tensorflow as tf
# import keras
# import numpy as np

# x = np.array([1,2,3,4,5,6])
# y = np.array([[1,2,3],[4,5,6]])
 
# print(x.shape)

from numpy import array, hstack

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence): #-1:
            break
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
        print(i)
        print("++++ seq_x ++++\n",sequence[i:end_ix, :-1],"\n++++++++++++++++++") 
        print("++++ seq_y ++++\n",seq_y,"\n++++++++++++++++++") 
        
        
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([12,25,35,45,55,65,75,85,95,105])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape) # (10, )
print(out_seq)

in_seq1 = in_seq1.reshape(len(in_seq1),1)
in_seq2 = in_seq2.reshape(len(in_seq2),1)
out_seq = out_seq.reshape(len(out_seq),1)


# print("in_seq1.shape: ",in_seq1.shape) # (10, 1)
# print(out_seq.shape) # (10, 1)

dataset = hstack((in_seq1,in_seq2,out_seq))
print("++++ dataset ++++\n",dataset,"\n++++++++++++++++++") 

n_steps = 3

x, y = split_sequence(dataset,n_steps)

print("++++ data x ++++\n",x,"\n++++++++++++++++++") 
print("++++ data y ++++\n",y,"\n++++++++++++++++++") 
print(len(x))

x = [1,2,3]
y= [4,5,6]
z=[x,y]
print(z)