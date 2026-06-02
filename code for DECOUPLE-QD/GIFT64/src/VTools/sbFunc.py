import numpy as np
# import config.config as config
# round_num=config.round_num

def globInd2info(bit_ind:int):
    rn=bit_ind//128
    ind=bit_ind%128
    SB_IND=-1
    SB_bit=-1
    if(ind>=64):#y bit
        ind=ind-64
        SB_IND=ind//4+rn*16
        SB_bit=ind%4+4
    else:
        SB_IND=ind//4+rn*16
        SB_bit=ind%4
    return SB_IND,SB_bit


def Info2GlobInd(SB_IND:int,SB_bit:int):
    rn=SB_IND//16
    sb=SB_IND%16 #the sbox index in current round
    res=-1
    if(SB_bit>=4):#y
        res=rn*128+64+sb*4+SB_bit-4
    else: #x
        res=rn*128+sb*4+SB_bit
    return res

if __name__=="__main__":
    print(Info2GlobInd(17,6))
    print(globInd2info(198))
