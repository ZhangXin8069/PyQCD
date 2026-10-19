
import numpy as np




Tensor4 = np.zeros((4, 4, 4, 4),dtype=int)

for i in range(0, 4):
    for j in range(0, 4):
        a = 0
        if i > j:
            a += 1.0
        for k in range(0, 4):
            b = 0
            if i > k:
                b += 1.0
            if j > k:
                b += 1.0
            for l in range(0, 4):
                c = 0
                if i > l:
                    c += 1.0
                if j > l:
                    c += 1.0
                if k > l:
                    c += 1.0
                if (a + b + c) % 2 == 0:
                    Tensor4[i, j, k, l] = 1
                elif (a + b + c) % 2 == 1:
                    Tensor4[i, j, k, l] = -1
                lst = [i, j, k, l]
                set_lst = set(lst)
                if len(set_lst) != len(lst):
                    Tensor4[i, j, k, l] = 0

Tensor3 = np.zeros((3, 3, 3),dtype=int)

for i in range(0, 3):
    for j in range(0, 3):
        a = 0
        if i > j:
            a += 1.0
        for k in range(0, 3):
            b = 0
            if i > k:
                b += 1.0
            if j > k:
                b += 1.0
            if (a + b) % 2 == 0:
                Tensor3[i, j, k] = 1
            elif (a + b) % 2 == 1:
                Tensor3[i, j, k] = -1
            lst = [i, j, k]
            set_lst = set(lst)
            if len(set_lst) != len(lst):
                Tensor3[i, j, k] = 0


sink_u_NN={1:{"s":"n","c":"b"},2:{"s":"r","c":"f"},3:{"s":"v","c":"h"},4:{"s":"z","c":"l"}}
sink_d_NN={1:{"s":"p","c":"d"},2:{"s":"x","c":"j"}}
source_u_NN={1:{"s":"m","c":"a"},2:{"s":"q","c":"e"},3:{"s":"u","c":"g"},4:{"s":"y","c":"k"}}
source_d_NN={1:{"s":"o","c":"c"},2:{"s":"w","c":"i"}}

NN_indexes=[]
NN_coeff=[]

uid=list(range(1,len(sink_u_NN)+1))
did=list(range(1,len(sink_d_NN)+1))
for srcuid1 in uid:
    leftuid2=[x for x in uid if x != srcuid1]
    for srcuid2 in leftuid2:
        leftuid3=[x for x in leftuid2 if x != srcuid2]
        for srcuid3 in leftuid3:
            leftuid4=[x for x in leftuid3 if x != srcuid3][0]
            for srcdid1 in did:
                leftdid2=[x for x in did if x != srcdid1][0]
                index= sink_u_NN[1]["s"]+source_u_NN[srcuid1]["s"]+sink_u_NN[1]["c"]+source_u_NN[srcuid1]["c"]  + ","  +\
                            sink_d_NN[1]["s"]+source_d_NN[srcdid1]["s"]+sink_d_NN[1]["c"]+source_d_NN[srcdid1]["c"]  + ","  +\
                                sink_u_NN[2]["s"]+source_u_NN[srcuid2]["s"]+sink_u_NN[2]["c"]+source_u_NN[srcuid2]["c"]  + ","  +\
                                    sink_u_NN[3]["s"]+source_u_NN[srcuid3]["s"]+sink_u_NN[3]["c"]+source_u_NN[srcuid3]["c"]  + ","  +\
                                        sink_d_NN[2]["s"]+source_d_NN[leftdid2]["s"]+sink_d_NN[2]["c"]+source_d_NN[leftdid2]["c"] + ","  +\
                                            sink_u_NN[4]["s"]+source_u_NN[leftuid4]["s"]+sink_u_NN[4]["c"]+source_u_NN[leftuid4]["c"] 
                
                if srcdid1==1:
                    coeff=Tensor4[srcuid1-1,srcuid2-1,srcuid3-1,leftuid4-1]*(-1)
                else:
                    coeff=Tensor4[srcuid1-1,srcuid2-1,srcuid3-1,leftuid4-1]
                NN_indexes.append(index)
                NN_coeff.append(coeff)



sink_u_NP={1:{"s":"n","c":"b"},2:{"s":"r","c":"f"},3:{"s":"v","c":"h"}}
sink_d_NP={1:{"s":"p","c":"d"},2:{"s":"x","c":"j"},3:{"s":"z","c":"l"}}
source_u_NP={1:{"s":"m","c":"a"},2:{"s":"q","c":"e"},3:{"s":"u","c":"g"}}
source_d_NP={1:{"s":"o","c":"c"},2:{"s":"w","c":"i"},3:{"s":"y","c":"k"}}

NP_indexes=[]
NP_coeff=[]

uid=list(range(1,len(sink_u_NP)+1))
did=list(range(1,len(sink_d_NP)+1))
for srcuid1 in uid:
    leftuid2=[x for x in uid if x != srcuid1]
    for srcuid2 in leftuid2:
        leftuid3=[x for x in leftuid2 if x != srcuid2][0]
        for srcdid1 in did:
            leftdid2=[x for x in did if x != srcdid1]
            for srcdid2 in leftdid2:
                leftdid3=[x for x in leftdid2 if x != srcdid2][0]
                index= sink_u_NP[1]["s"]+source_u_NP[srcuid1]["s"]+sink_u_NP[1]["c"]+source_u_NP[srcuid1]["c"]  + ","  +\
                            sink_d_NP[1]["s"]+source_d_NP[srcdid1]["s"]+sink_d_NP[1]["c"]+source_d_NP[srcdid1]["c"]  + ","  +\
                                sink_u_NP[2]["s"]+source_u_NP[srcuid2]["s"]+sink_u_NP[2]["c"]+source_u_NP[srcuid2]["c"]  + ","  +\
                                    sink_u_NP[3]["s"]+source_u_NP[leftuid3]["s"]+sink_u_NP[3]["c"]+source_u_NP[leftuid3]["c"]  + ","  +\
                                        sink_d_NP[2]["s"]+source_d_NP[srcdid2]["s"]+sink_d_NP[2]["c"]+source_d_NP[srcdid2]["c"] + ","  +\
                                            sink_d_NP[3]["s"]+source_d_NP[leftdid3]["s"]+sink_d_NP[3]["c"]+source_d_NP[leftdid3]["c"]  
                
                coeff=-1*Tensor3[srcuid1-1,srcuid2-1,leftuid3-1]*Tensor3[srcdid1-1,srcdid2-1,leftdid3-1]
                NP_indexes.append(index)
                NP_coeff.append(coeff)