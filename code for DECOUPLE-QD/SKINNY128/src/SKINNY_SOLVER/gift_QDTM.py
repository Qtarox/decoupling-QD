import numpy as np
SBOX=[1,10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14]
def xddt_list(input,output):
    if(input==0 and output==0):
        return [i for i in range(16)]
    res=[]
    for x in range(16):
        x1=x
        x2=x^input
        y1=SBOX[x1]
        y2=SBOX[x2]
        if(y1^y2==output and input!=0):
            res.append(x)
            #print("XDDT("+str(input)+", "+str(output)+")="+str(tmp))
    return res

def inner(c,d):
    return bin(c & d).count('1') % 2


def get_QDTM(a,b,X_lst):
    res=np.zeros((16,16),dtype=int)
    if(X_lst is None or X_lst==[]):
        x_lst=xddt_list(a,b)
    else:
        x_lst=X_lst
    non_zero=0
    for u in range(16):
        for v in range(16):
            sum=0
            for x in x_lst:
                sum+=(-1)**(inner(u,x)^inner(v,SBOX[x]))
            res[u][v]=sum
            if(sum!=0):
                non_zero+=1
    return res
    

if __name__ == "__main__":
    a=0
    b=0
    print(get_QDTM(a,b))
    print(get_QDTM(a,b)[1][8])
    print(get_QDTM(a,b)[8][4])
    print(get_QDTM(a,b)[8][5])
    # print(get_QDTM(a,b)[10][11])
