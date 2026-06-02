import numpy as np
from coreFunc.CHAR2DIC import *
import config.config as config
x_dic1=load_dic(config.file_path+"act_x.json")
y_dic1=load_dic(config.file_path+"act_y.json")
active_bit_dic=load_dic(config.file_path+"act_bit.json")
import os

# Define the name of the new folder


def creat_dic_GIFT(file_path,round):
    x_dic={}
    y_dic={}
    print("running!")
    for r in range(len(round)):
        for x_index in range(16):
            if(round[r][0][x_index]==0):
                continue
            x_tmp='x_'+str(r)+'_'+str(x_index)
            y_tmp='y_'+str(r)+'_'+str(x_index)
            l_x=xddt_list(round[r][0][x_index],round[r][1][x_index])
            l_y=yddt_list(round[r][0][x_index],round[r][1][x_index])
            x_dic[x_tmp]=l_x.copy()
            y_dic[y_tmp]=l_y.copy()
    # original_stdout = sys.stdout
    print("x_dic: ",x_dic)
    # Specify the file name where you want to save the output
    file_x= file_path+"act_x.json"

        # Open the file in write mode, this will create the file if it doesn't exist
    with open(file_x, 'w') as json_file1:
        # Redirect the standard output to the file
        json.dump(x_dic, json_file1)
    
    file_y= file_path+"act_y.json"

        # Open the file in write mode, this will create the file if it doesn't exist
    with open(file_y, 'w') as json_file2:
        # Redirect the standard output to the file
        json.dump(y_dic, json_file2)
def create_folder(pth):
# Create the new folder if it doesn't already exist
    if not os.path.exists(pth):
        os.makedirs(pth)
    # Verify if the folder has been created
    os.path.exists(pth)

def is_in_active_dic(num,active_bit_dic):
    
    if(str(num) in active_bit_dic):
        return True
    else:
        return False

def get_active_bit(x_dic,y_dic):
    pattern = r"x_(\d+)_([\d]+)"
    res={}
    for key in x_dic:
        match = re.match(pattern, key)
        X_lst=x_dic[key]
        rn=int(match.group(1))
        ind=int(match.group(2))#sb_ind
        for i in range(4):# check all 4 bits
            # print(key)
            activeFlg=True
            initial=X_lst[0]>>i&1
            for x in X_lst:
                curBit=x>>i&1
                if(initial!=curBit):
                    activeFlg=False
                    break
            if(activeFlg):    
                res[(rn*128+ind*4+i)]=initial
    pattern = r"y_(\d+)_([\d]+)"
    for key in y_dic:
        match = re.match(pattern, key)
        Y_lst=y_dic[key]
        rn=int(match.group(1))
        ind=int(match.group(2))
        for i in range(4):# check all 4 bits
            activeFlg=True
            initial=Y_lst[0]>>i&1
            for y in Y_lst:
                curBit=y>>i&1
                if(initial!=curBit):
                    activeFlg=False
                    break
            if(activeFlg):    
                res[rn*128+64+ind*4+i]=initial

    return res
def show_FSB(F_SB):
    for i in F_SB:
        print('SB '+str(i)+": [",end='')
        for b in F_SB[i]:
            if(b==-1):
                print("*, ",end='')
            else:
                print(b,", ",end='')
        print("]")
def is_bit_active(x_list,bit_ind):
    sum=0
    for i in x_list:
        tmp=i//(2**bit_ind)
        b=tmp%2
        sum=sum+b
    if(sum==0 or sum==len(x_list)):
        return True
    return False
def get_bit_val(list_t, ind):
    # print(list_t)
    ele=list_t[0]
    ele=ele//(2**ind)
    bit=ele%2
    return bit
def x_info2(rn, SB_ind,active_dic,x_dic):
    X_list=[-1,-1,-1,-1]
    strt_ind=rn*128+SB_ind*4
    for i in range(4):
        tmp_x_ind=strt_ind+i
        try:
            value=active_dic[str(tmp_x_ind)]
            # print(tmp_x_ind)
            X_list[i]=value
        except:
            # print(tmp_x_ind)
            pass
    try:
        x_lst=x_dic["x_"+str(rn)+"_"+str(SB_ind)]
        print()
        print(x_lst)
        for i in range(4):
            for j in range(i+1,4):
                eq_flg=0
                neq_flg=0
                for x in x_lst:
                    x_val=int(x)
                    bit_i=(x_val>>i)&1
                    bit_j=(x_val>>j)&1
                    if(bit_i==bit_j):
                        eq_flg+=1
                    if(bit_i!=bit_j):
                        neq_flg+=1
                if(eq_flg==len(x_lst)):
                    if(X_list[i]==-1):
                        X_list[i]=2
                        X_list[j]=2
                    elif(X_list[j]==-1):
                        X_list[j]=X_list[i]
                if(neq_flg==len(x_lst)):
                    if(X_list[i]==-1):
                        X_list[i]=2
                        X_list[j]=X_list[i]^1
                    elif(X_list[j]==-1):
                        X_list[j]=X_list[i]^1   
    except:
        pass     

    return X_list.copy()

def x_info(x_list):
    res=[]
    for i in range(4):
       tmp_bit=(x_list[0]//(2**i))%2
       flg=True
       for x in x_list:
            if(tmp_bit==(x//(2**i))%2):
                pass
            else:
                res.append(-1)
                flg=False
                break
       if(flg):
           res.append(tmp_bit)
    return res

if __name__=="__main__":
    x_l=[8,13]
    print(x_info(x_l))