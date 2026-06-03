
import numpy as np
Sbox=[ 12 , 6 , 9 , 0 , 1 , 10 , 2 , 11 , 3 , 8 , 5 , 13 , 4 , 14 , 7 , 15 ]
M_EQ=np.load("./M_EQ.npy")
def key_schedule(round=0,key_index=0):
    key_permu=[9 , 15 , 8 , 13 , 10 , 14 , 12 , 11 , 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7]
    tmp=key_index
    for i in range(round):
        tmp=key_permu[tmp]
    return tmp   
def Global_mat(res,M_EQ,round_num):
    for i in range(np.shape(res)[0]):
        equ_num=i%16
        rn=i//16 #the equ_num th equation in rn round
        for k in range(40):
            if(M_EQ[equ_num][k]==1):
                if(k<16):#x_r+1_k
                    res[i][(rn+1)*32+k]=1
                elif(k<32):#y_r_k
                    res[i][rn*32+k]=1
                else:# is key index
                    rn_k_ind=k-32
                    break
        k_ind=key_schedule(rn,rn_k_ind)
        res[i][(32)*(round_num+1)+k_ind]=1
    return res

def Global_mat_bit( round_num):
    res=np.zeros((64*round_num,128 * (round_num + 1) + 64),dtype=int)
    # 此时 res 的行数应该是 round_num * 64 (因为每轮有 16个cell * 4比特 = 64个方程)
    for i in range(np.shape(res)[0]):
        equ_num_bit = i % 64            # 当前轮的第几个 bit 方程 (0~63)
        rn = i // 64                    # 当前是第 rn 轮
        
        equ_num_cell = equ_num_bit // 4 # 对应 M_EQ 里的第几个 cell 方程 (0~15)
        bit_idx = equ_num_bit % 4       # 对应 cell 内部的第几个 bit (0~3)
        
        for k in range(40):
            if(M_EQ[equ_num_cell][k] == 1):
                if(k < 16): 
                    # x_r+1_k (下一轮状态)
                    # 原本每个 round 占 32 个 cell，现在占 32 * 4 = 128 个 bit
                    res[i][(rn + 1) * 128 + k * 4 + bit_idx] = 1
                
                elif(k < 32): 
                    # y_r_k (当前轮状态)
                    # k 在 16~31 之间，乘以 4 后映射到对应的 bit 位
                    res[i][rn * 128 + k * 4 + bit_idx] = 1
                
                else: 
                    # 密钥/Tweakey 索引
                    rn_k_ind = k - 32
                    k_ind = key_schedule(rn, rn_k_ind)
                    # 密钥变量放在所有状态变量的最后
                    # 起点是 128 * (round_num + 1)
                    res[i][128 * (round_num + 1) + k_ind * 4 + bit_idx] = 1
                    break 
    return res

def show_L_equ_GIFT(lmat,active_bit_dic,round_num,FILE_FLG=False,file=None):
    if(not file):
       F="output.txt"
    else:
        F=file
    L=""
    if(FILE_FLG):
        with open(F, "w") as file1:
            for i in range(np.shape(lmat)[0]):
                # print(i)
                l_tmp=""
                flag=False
                for j in range(128*(round_num+1)):
                    rn=j//128
                    ind=j%128
                    if((str(j) in active_bit_dic) and lmat[i][j]==1):
                        if(ind<64):
                            l_tmp=l_tmp+" + [x_"+str(int(rn))+"_"+str(ind)+"]"
                        else:
                            l_tmp=l_tmp+" + [y_"+str(int(rn))+"_"+str((ind-64))+"]"
                    elif(lmat[i][j]==1):
                        if(ind<64):
                            l_tmp=l_tmp+" + x_"+str(int(rn))+"_"+str(ind)+""
                        else:
                            l_tmp=l_tmp+" + y_"+str(int(rn))+"_"+str((ind-64))+""

                flag=False
                for k in range(64):
                    if(lmat[i][k+round_num*128+128]==1):
                        # print(k+round_num*128)
                        flag=True
                        l_tmp=l_tmp+' + k_'+str(k)+" = 0"
                if(flag==False):
                    l_tmp=l_tmp+'= 0   '
                l_tmp+='  SBOX: '+str(lmat[i][-1])
                print(l_tmp)
                print(l_tmp, file=file1)
                L+=l_tmp+'\n'
    else:
        for i in range(np.shape(lmat)[0]):
            # print(i)
            l_tmp=""
            flag=False
            for j in range(128*(round_num+1)):
                rn=j//128
                ind=j%128
                if((str(j) in active_bit_dic) and lmat[i][j]==1):
                    if(ind<64):
                        l_tmp=l_tmp+" + [x_"+str(int(rn))+"_"+str(ind)+"]"
                    else:
                        l_tmp=l_tmp+" + [y_"+str(int(rn))+"_"+str((ind-64))+"]"
                elif(lmat[i][j]==1):
                    if(ind<64):
                        l_tmp=l_tmp+" + x_"+str(int(rn))+"_"+str(ind)+""
                    else:
                        l_tmp=l_tmp+" + y_"+str(int(rn))+"_"+str((ind-64))+""

            flag=False
            for k in range(64):
                if(lmat[i][k+round_num*128+128]==1):
                    # print(k+round_num*128)
                    flag=True
                    l_tmp=l_tmp+' + k_'+str(k)+" = 0"
            if(flag==False):
                l_tmp=l_tmp+'= 0   '
            
            print(l_tmp)
            L+=l_tmp+'\n'
    return L



def show_L_equ_GIFT_extract(lmat,round_num,FILE_FLG=False,file=None):
    if(not file):
       F="output.txt"
    else:
        F=file
    if(FILE_FLG):
        with open(F, "w") as file1:
            for i in range(np.shape(lmat)[0]):
                # print(i)
                l_tmp=""
                flag=False
                for j in range(128*(round_num+1)):
                    rn=j//128
                    ind=j%128
                    if( lmat[i][j]==2):
                        if(ind<64):
                            l_tmp=l_tmp+" + [x_"+str(int(rn))+"_"+str(ind)+"]"
                        else:
                            l_tmp=l_tmp+" + [y_"+str(int(rn))+"_"+str((ind-64))+"]"
                    elif(lmat[i][j]==1):
                        if(ind<64):
                            l_tmp=l_tmp+" + x_"+str(int(rn))+"_"+str(ind)+""
                        else:
                            l_tmp=l_tmp+" + y_"+str(int(rn))+"_"+str((ind-64))+""

                flag=False
                for k in range(128):
                    if(lmat[i][k+round_num*128+128]==1):
                        # print(k+round_num*128)
                        flag=True
                        l_tmp=l_tmp+' + k_'+str(k)+" = 0"
                if(flag==False):
                    l_tmp=l_tmp+'= 0   '

                print(l_tmp)
                print(l_tmp, file=file1)
    else:
        for i in range(np.shape(lmat)[0]):
            # print(i)
            l_tmp=""
            flag=False
            for j in range(128*(round_num+1)):
                rn=j//128
                ind=j%128
                if(lmat[i][j]==2):
                    if(ind<64):
                        l_tmp=l_tmp+" + [x_"+str(int(rn))+"_"+str(ind)+"]"
                    else:
                        l_tmp=l_tmp+" + [y_"+str(int(rn))+"_"+str((ind-64))+"]"
                elif(lmat[i][j]==1):
                    if(ind<64):
                        l_tmp=l_tmp+" + x_"+str(int(rn))+"_"+str(ind)+""
                    else:
                        l_tmp=l_tmp+" + y_"+str(int(rn))+"_"+str((ind-64))+""

            flag=False
            for k in range(64):
                if(lmat[i][k+round_num*128+128]==1):
                    # print(k+round_num*128)
                    flag=True
                    l_tmp=l_tmp+' + k_'+str(k)+" = 0"
            if(flag==False):
                l_tmp=l_tmp+'= 0   '

            print(l_tmp)

if __name__=="__main__":
    res=Global_mat_bit(2)
    show_L_equ_GIFT_extract(res,2)