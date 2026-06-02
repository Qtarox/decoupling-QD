import numpy as np
import config.config as config

round_num= config.round_num
from coreFunc.CHAR2DIC import load_dic
active_bit_dic=load_dic(config.file_path+"act_bit.json")
var_lst=["x0","x1","x2","x3","x0x1","x0x2","x0x3","x1x2","x1x3","x2x3","x0x2x3","x1x2x3"]

def show_L_equ_GIFT(lmat,FILE_FLG=False,file=None):
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
                    l_tmp=l_tmp+' + k_'+str(k)+" = 0"
            if(flag==False):
                l_tmp=l_tmp+'= 0   '
            l_tmp+='  SBOX: '+str(lmat[i][-1])
            print(l_tmp)




# def show_L_equ_GIFT(lmat):
#     for i in range(np.shape(lmat)[0]):
#         l_tmp=""
#         flag=False
        
#         for j in range(128*round_num+128):
#             rn=j//128
#             ind=j%128
#             if(flag==False and lmat[i][j]==1):
#                 if(ind<64):
#                     l_tmp=l_tmp+" x_"+str(int(rn))+"_"+str(ind)
#                 else:
#                     l_tmp=l_tmp+" y_"+str(int(rn))+"_"+str((ind-64))
#                 flag=True
#             elif(flag==True and lmat[i][j]==1):
#                 if(ind<64):
#                     l_tmp=l_tmp+" + x_"+str(int(rn))+"_"+str(ind)
#                 else:
#                     l_tmp=l_tmp+" + y_"+str(int(rn))+"_"+str((ind-64))
#         flag=False
#         for k in range(32*round_num):
#             if(lmat[i][k+round_num*128+128]==1):
#                 flag=True
#                 l_tmp=l_tmp+' + k_'+str(k//32)+"_"+str(k%32)+" = 0"
#         if(flag==False):
#             l_tmp=l_tmp+'= 0'
#         print(l_tmp)
def show_linear_mat(lmat):
    for i in range(np.shape(lmat)[0]):
        print("equ "+str(i), end=" : ")
        l_tmp=""
        flag=False
                
        for j in range(np.shape(lmat)[1]-128):
            rn=j//128
            ind=j%128
            if(lmat[i][j]==1):
                if((ind)<64):#if is x linear var
                    l_tmp=l_tmp+" + x_"+str(int(rn))+"_"+str(ind)+""
                else:
                    l_tmp=l_tmp+" + y_"+str(int(rn))+"_"+str((ind-64))+""

        for k in range(128):
            if(lmat[i][k+np.shape(lmat)[1]-128]==1):
                # print(k+round_num*128)
                l_tmp=l_tmp+' + k_'+str(k)
                
        print(l_tmp+" = 0 ")

# def show_L_equ_GIFT3(lmat):
    
    for i in range(np.shape(lmat)[0]):
        print("equ "+str(i), end=" : ")
        l_tmp=""
        flag=False
        for j in range(256*(round_num+1)):
            rn=j//256
            ind=j%256
            if(lmat[i][j]==1):
                if(ind<192 and (ind%12)<4):#if is x linear var
                    l_tmp=l_tmp+" + x_"+str(int(rn))+"_"+str((ind//12)*4+ind%12)+""
                elif(ind>=192):
                    l_tmp=l_tmp+" + y_"+str(int(rn))+"_"+str((ind-192))+""
                else:
                    l_tmp=l_tmp+"+ "+var_lst[ind%12]
        if(np.shape(lmat)[1]>256*(round_num+1)):

            for k in range(128):
                if(lmat[i][k+round_num*256+256]==1):
                    # print(k+round_num*128)
                    l_tmp=l_tmp+' + k_'+str(k)

            l_tmp=l_tmp+'= 0   '
        else:
            l_tmp=l_tmp+'= 0   '
        print(l_tmp)


# def show_L_equ_GIFT4(lmat):
    var_lst2=["x0x1","x0x2","x0x3","x1x2","x1x3","x2x3","x0x2x3","x1x2x3"]
    for i in range(np.shape(lmat)[0]):
        print("equ "+str(i), end=" : ")
        l_tmp=""
        for j in range(256*(round_num+1)):
            if(j<128*(round_num+1)):#linear x and y
                rn=j//128
                ind=j%128
                if(lmat[i][j]==1):
                    if(ind<64):#if is x linear var
                        l_tmp=l_tmp+" + x_"+str(int(rn))+"_"+str(ind)+""
                    else:
                        l_tmp=l_tmp+" + y_"+str(int(rn))+"_"+str((ind-64))+""
            elif(j<256*(round_num+1)):#nonlinear x


                rn=(j-128*(round_num+1))//128
                ind=((j-128*(round_num+1))%128)
                if(lmat[i][j]==1):
                    l_tmp=l_tmp+"+ "+var_lst2[ind%8]
        if(np.shape(lmat)[1]>256*(round_num+1)):

            for k in range(128):
                if(lmat[i][k+round_num*256+256]==1):
                    # print(k+round_num*128)
                    l_tmp=l_tmp+' + k_'+str(k)

            l_tmp=l_tmp+'= 0   '
        else:
            l_tmp=l_tmp+'= 0   '
        print(l_tmp)
        
def show_res(mat,res_lst):
    cnt=0
    for sub_lst in res_lst:
        select=mat[sub_lst]
        print("constraint ",cnt," :")
        cnt+=1
        show_L_equ_GIFT(select)
        