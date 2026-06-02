import numpy as np
import config.config as config
from VTools.TOOL import create_folder,is_strict_active_GIFT
round_num=config.round_num
def prune_mat(G_mat):
    SB_num=16*(round_num-1)

    Gmat=G_mat[:,:128*(round_num+1)].copy()# prune keys
    P3=[]
    for i in range(np.shape(G_mat)[0]):
        for j in range(128*(round_num+1)):
            if(is_strict_active_GIFT(j)):
                G_mat[i][j]=0
        if(i<64*round_num):
            pass
        elif(i<64*round_num+SB_num*15):
            matp2=G_mat[64*round_num:i][:]
            if(np.all(G_mat[i]==0) or np.any(np.all(matp2 == G_mat[i], axis=1))):
                for j in range(np.shape(G_mat)[1]):
                    G_mat[i][j]=3
        else: 
            matp3=G_mat[:64*round_num+SB_num*15][:]
            if(np.all(G_mat[i]==0) or np.any(np.all(matp3 == G_mat[i], axis=1))):
                pass
            else:
                P3.append(i)
    print("len:", len(P3),"P3: ", P3)           
    Final_mat=np.zeros((64*round_num+SB_num*15+len(P3),128*(round_num+1)),dtype=int)
    for i in range(np.shape(Final_mat)[0]):
        if(i<64*round_num+SB_num*15):
            Final_mat[i][:]=Gmat[i][:]
        else:
            Final_mat[i][:]=Gmat[P3[i-(64*round_num+SB_num*15)]][:]
                       
    print(np.shape(Final_mat),"P2 IND: ",64*round_num+SB_num*15)
    return Final_mat


def get_linear(MAT):
    # remain the key
    #MAT shape has been pruned, the first 4 and last 4 for every 16 elements are linear x and y
    # we only extract the linear x, y column
    print(np.shape(MAT))
    column_list=[]
    for j in range(np.shape(MAT)[1]):
        if(j>=256*(round_num+1)):#is key
            column_list.append(j)
        elif(j<128*(round_num+1)):#is linear x,y
            column_list.append(j)
    res=MAT[:,column_list].copy()
    return res