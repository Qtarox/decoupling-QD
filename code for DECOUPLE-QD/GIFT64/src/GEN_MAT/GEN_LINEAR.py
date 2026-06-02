import numpy as np
GIFT_P=[0, 17, 34, 51, 48,  1, 18, 35, 32, 49,  2, 19, 16, 33, 50,  3,
        4, 21, 38, 55, 52,  5, 22, 39, 36, 53,  6, 23, 20, 37, 54,  7,
        8, 25, 42, 59, 56,  9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11,
        12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15]

GIFT_INV=[ 0,  5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63,
            12,  1,  6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59,
            8, 13,  2,  7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55,
            4,  9, 14,  3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51]

Sbox=[1,10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14]

#################################### MAPPING AND KEY-SCHEDULE #####################################################
def map_array():
    MAP=[]
    for i in range(64):
        map_i=4*(i//16)+16*((3*((i%16)//4)+i%4)%4)+i%4
        MAP.append(map_i)
    P_MAP=np.array(MAP)
    revP_MAP=np.zeros(np.shape(P_MAP),dtype=int)
    for i in range(64):
        revP_MAP[P_MAP[i]]=i
    return P_MAP,revP_MAP

def key_schedule(rn):
    l6=[12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11]
    l7=[18,19,20,21,22,23,24,25,26,27,28,29,30,31,16,17]
    MAP_key=[]
    for i in range(96):
        MAP_key.append(32+i)
    for i in range(16):
        MAP_key.append(l6[i])
    for i in range(16):
        MAP_key.append(l7[i])
    # print(MAP_key)
    Tmp_Key=list(range(128))
    Tmp_Key2=list(range(128))
    for i in range(rn):
        Tmp_Key2=Tmp_Key.copy()
        for j in range(128):
            Tmp_Key2[j]=Tmp_Key[MAP_key[j]]
        Tmp_Key=Tmp_Key2.copy()
    # print(Tmp_Key)
    U=Tmp_Key2[16:32]
    V=Tmp_Key2[0:16]
    return U,V

def key_sch_mat(gmat,round_num):# reform the gmat into key schedule
    G_mat=np.zeros((np.shape(gmat)[0],128*(round_num+1)+128+1),dtype=int)
    for i in range(np.shape(G_mat)[0]):
        for j in range(np.shape(gmat)[1]-1):
            if(gmat[i][j]==1):
                if(j<128*(round_num+1)):
                    G_mat[i][j]=1
                else:
                    key=j-128*(round_num+1)
                    k_rn=key//32
                    k_ind=key%32
                    U,V=key_schedule(k_rn)
                    if(k_ind%2==0):
                        G_mat[i][128*(round_num+1)+V[k_ind//2]]=1
                    else:
                        G_mat[i][128*(round_num+1)+U[k_ind//2]]=1
        G_mat[i][-1]=gmat[i][-1]
    return G_mat


def get_para(key):
    rn=''
    ind=''
    cnt=0
    rn_flg=False
    ind_flg=False
    for i in range(len(key)):
        if(ind_flg):
            if(key[i]!=''):
                ind=ind+key[i]
        if(rn_flg):
            if(key[i]!='_'):
                rn=rn+key[i]
            else:
                rn_flg=False
                ind_flg=True
        if(key[i]=='_' and cnt==0):
            rn_flg=True
            cnt=1

    return int(rn),int(ind)

##########################################################################
def genLinear(R):
    L_mat=np.zeros((64*(R),128*(R+1)+32*(R+1)+1),dtype=int)
    for i in range(np.shape(L_mat)[0]):
        round=i//64
        ind=i%64# original ind of x
        ind_x=ind      
        L_mat[i][(round+1)*128+ind_x]=1#x_r_i
        L_mat[i][round*128+64+GIFT_INV[ind_x]]=1 #y_(r)_j
        if((ind%4)<2):
            k_ind=(ind//4)*2+ind%4
            # print(R,round,k_ind)
            L_mat[i][128*(R+1)+32*round+k_ind]=1
        L_mat[i][-1]=i

    return key_sch_mat(L_mat,R)

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
                for k in range(128):
                    if(lmat[i][k+round_num*128+128]==1):
                        # print(k+round_num*128)
                        flag=True
                        l_tmp=l_tmp+' + k_'+str(127-k)+" = 0"
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
            for k in range(128):
                if(lmat[i][k+round_num*128+128]==1):
                    # print(k+round_num*128)
                    flag=True
                    l_tmp=l_tmp+' + k_'+str(127-k)+" = 0"
            if(flag==False):
                l_tmp=l_tmp+'= 0   '
            
            print(l_tmp)
            L+=l_tmp+'\n'
    return L

def show_L_equ_GIFT_ectract(lmat,round_num,FILE_FLG=False,file=None):
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
                l_tmp+='  SBOX: '+str(lmat[i][-1])
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
            for k in range(128):
                if(lmat[i][k+round_num*128+128]==1):
                    # print(k+round_num*128)
                    flag=True
                    l_tmp=l_tmp+' + k_'+str(k)+" = 0"
            if(flag==False):
                l_tmp=l_tmp+'= 0   '
            l_tmp+='  SBOX: '+str(lmat[i][-1])
            print(l_tmp)

if __name__=="__main__":
    L_mat=genLinear(3)       
    show_L_equ_GIFT(L_mat,3)