import numpy as np
active_bit_dic={}
def show_L_equ_GIFT(lmat,round_num,FILE_FLG=False,file=None):
    if(not file):
       F="output.txt"
    else:
        F=file
    if(FILE_FLG):
        with open(F, "w") as file1:
            for i in range(np.shape(lmat)[0]):
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
                        flag=True
                        l_tmp=l_tmp+' + k_'+str(k)+" = 0"
                if(flag==False):
                    l_tmp=l_tmp+'= 0   '
                l_tmp+='  SBOX: '+str(lmat[i][-1])
                print(l_tmp)
                print(l_tmp, file=file1)
    else:
        for i in range(np.shape(lmat)[0]):
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
                    flag=True
                    l_tmp=l_tmp+' + k_'+str(k)+" = 0"
            if(flag==False):
                l_tmp=l_tmp+'= 0   '
            l_tmp+='  SBOX: '+str(lmat[i][-1])
            print(l_tmp)
def show_res(mat,res_lst):
    cnt=0
    for sub_lst in res_lst:
        select=mat[sub_lst]
        print("constraint ",cnt," :")
        cnt+=1
        show_L_equ_GIFT(select)
