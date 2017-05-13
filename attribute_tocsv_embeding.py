#!usr/bin/env Python
#coding:utf-8

from numpy import *
import numpy as np
import random
import csv
"""
feature choose,a 25d vector indicate the number of ACDEFGHIKLMNPQRSTVWYcnm(DN)(AN),
23d vector indicate the animo acid,a window of cleavage 4,4*23+2,2 indicate N-term modification,
4d vector indicate the charge of peptide sequence.
"""

dicB={'A':206.4,'c':206.2,'D':208.6,'E':215.6,'F':212.1,\
      'G':202.7,'H':223.7,'I':210.8,'K':221.8,'L':209.6,\
      'M':213.3,'N':212.8,'P':214.4,'Q':214.2,'R':237.0,\
      'S':207.6,'T':211.7,'V':208.7,'W':216.1,'Y':213.1}
dicM={'A':71.03711,'c':160.03065,'D':115.02694,'E':129.04259,'F':147.06841,\
      'G':57.02146,'H':137.05891,'I':113.08406,'K':128.09496,'L':113.08406,\
      'M':131.04048,'N':114.04293,'P':97.05276,'Q':128.05858,'R':156.10111,\
      'S':87.03203,'T':101.04768,'V':99.06841,'W':186.07931,'Y':163.06332}
dicS={'A':0,'c':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,\
      'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,\
      'S':15,'T':16,'V':17,'W':18,'Y':19}
dicHe={'A':1.24,'c':0.79,'D':0.89,'E':0.85,'F':1.26,'G':1.15,'H':0.97,\
       'I':1.29,'K':0.88,'L':1.28,'M':1.22,'N':0.94,'P':0.57,'Q':0.96,\
       'R':0.95,'S':1.00,'T':1.09,'V':1.27,'W':1.07,'Y':1.11}
dicHy={'A':0.16,'c':2.50,'D':-2.49,'E':-1.50,'F':5.00,'G':-3.31,'H':-4.63,'I':4.41,
      'K':-5.00,'L':4.76,'M':3.23,'N':-3.79,'P':-4.92,'Q':-2.76,'R':-2.77,'S':-2.85,
      'T':-1.08,'V':3.02,'W':4.88,'Y':2.00}


with open('data/data_embeding.csv','w',newline='') as wf:
    writer=csv.writer(wf)
    CountOfAAInPeptide=''
    CountOfAAInYion=''
    CountOfAAInBion=''
    IsolationWidth=''
    C_Identity=''
    N_Identity=''
    for i in range(1,21):
        CountOfAAInPeptide+='CountOfAAInPeptide_'+str(i)+','
        CountOfAAInYion+='CountOfAAInYion_'+str(i)+','
        CountOfAAInBion+='CountOfAAInBion_'+str(i)+','
        C_Identity+='C_Identity_'+str(i)+','
        N_Identity+='N_Identity_'+str(i)+','
    for i in range(4):
        for j in range(1,21):
            IsolationWidth+='IsolationWidth_'+str(i+1)+"_"+str(j)+','
    writeStr='Number,Peptide,'+CountOfAAInPeptide+CountOfAAInYion+CountOfAAInBion+IsolationWidth+C_Identity+N_Identity+'IsTheSamePort,RatioOfyAndPeptide,RatioOfbAndPeptide,y_Distance,b_Distance,R_Basicity,L_Basicity,R_Helicity,L_Helicity,R_Hydrophobicity,L_Hydrophobicity,y_Basic,b_Basic,y_Helicity,b_Helicity,y_Hydrophobicity,b_Hydrophobicity,AvgOfBasic,DvalueOfBasic,AvgOfHelicity,DvalueOfHelicity,AvgOfHydrophobicity,DvalueOfHydrophobicity,y_Mass,b_Mass,DvalueOfMcratioOfyAndPep,CountOfBasicAA,y_CountOfBasicAA,b_CountOfBasicAA,LengthOfPeptide,P_Basic,P_Helicity,P_Hydrophobicity,McRatioOfPeptide,Intensity'
    writer.writerow(writeStr.split(','))
    cunt=0
    f = open('data/ion2_y.txt','r')
    data = [];pep_mz = 0;pep = '1';pep_b = 0;pep_he = 0;pep_hy=0
    for line in f:
        flag=False
        try:
            attribute = zeros(214)
            f_line = line.split('\t')
            peptide = list(f_line[0])
            ion = list(f_line[2])
            i = 0
            ion_format = f_line[5]
            #肽中每种氨基酸出现的次数 0~19
            for key in dicS.keys():
                if key in peptide:
                    attribute[dicS[key]] = peptide.count(key)

            #碎裂窗口，size=2
            ion_format = f_line[5]
            if len(ion) > 2 and len(ion) < len(peptide) - 2 and ion_format[0] == 'y':
                j = 0
                for i in range(len(peptide) - len(ion) - 2,len(peptide) - len(ion) + 2):
                    attribute[60 + 20*j+dicS[peptide[i]]] = 1
                    j += 1
            elif len(ion) > 2 and len(ion) < len(peptide) - 2 and ion_format[0] == 'b':
                j = 0
                for i in range(len(ion) - 2,len(ion) + 2):
                    attribute[60 + j*20+dicS[peptide[i]]] = 1
                    j += 1
            elif (len(ion) <= 2 and ion_format[0] == 'y') or (len(ion) >= len(peptide) - 2 and ion_format[0] == 'b'):
                j = 0
                if len(ion) == 1 or len(ion) == len(peptide) - 1:
                    for i in range(len(peptide) - 3,len(peptide)):
                        attribute[60 + 20*j + dicS[peptide[i]]] = 1
                        j += 1
                elif len(ion) == 2 or len(ion) == len(peptide) - 2:
                    for i in range(len(peptide) - 4,len(peptide)):
                        attribute[60 + 20*j + dicS[peptide[i]]] = 1
                        j += 1
            elif (len(ion) <= 2 and ion_format[0] == 'b') or (len(ion) >= len(peptide) - 2 and ion_format[0] == 'y'):
                if len(ion) == 1 or len(ion) == len(peptide) - 1:
                    j = 2
                    for i in range(3):
                       attribute[60 + j*20+dicS[peptide[i]]] = 1
                       j += 1
                elif len(ion) == 2 or len(ion) == len(peptide) - 2:
                    j = 1
                    for i in range(4):
                       attribute[60 + j*20+dicS[peptide[i]]] = 1
                       j += 1

            #b/y碎片离子中每种氨基酸出的次数
            #y离子，20~39
            for key in dicS.keys():
                if key in ion:
                    attribute[20 + dicS[key]] = ion.count(key)

            if ion_format[0] == 'y':        
                other_ion = peptide[:len(peptide) - len(ion)]
            else:
                other_ion = peptide[len(ion):]
            for key in dicS.keys():
                if key in other_ion:
                    attribute[40 + dicS[key]] = other_ion.count(key)
                    
            #C/N端肽的身份
            attribute[140 + dicS[peptide[0]]] = 1
            attribute[160 + dicS[peptide[-1]]] = 1

            #碎裂点是否在肽的一端
            if len(ion) == 1:
                attribute[180] = 1

            #计算肽的质量、碱性、疏水性、螺旋性
            if pep != peptide:
                pep_mz = 20.0;pep_b = 0.0;pep = peptide;pep_he = 0;pep_hy=0
                for i in range(len(peptide)):
                    pep_mz += dicM[peptide[i]]
                    pep_b += dicB[peptide[i]]
                    pep_he += dicHe[peptide[i]]
                    pep_hy += dicHy[peptide[i]]

            #计算碎片离子的质量、碱性、疏水性、螺旋性
            ion_mz = 19.0;ion_b = 0.0;ion_he=0;ion_hy=0
            for i in range(len(ion)):
                ion_mz += dicM[ion[i]]
                ion_b += dicB[ion[i]]
                ion_he += dicHe[ion[i]]
                ion_hy += dicHy[ion[i]]

            if ion_format[0] == 'y':
                #b/y离子的质量与肽质量的比
                attribute[181] = ion_mz / pep_mz
                attribute[182] = 1 - attribute[181]
                
                #碎裂点距肽C/N端的距离
                attribute[184] = len(ion)
                attribute[183] = len(peptide) - len(ion)
                
                #碎裂点两个氨基酸的碱性、螺旋性、疏水性
                attribute[185] = dicB[peptide[-len(ion)]]
                attribute[186] = dicB[peptide[-len(ion) - 1]]
                attribute[187] = dicHe[peptide[-len(ion)]]
                attribute[188] = dicHe[peptide[-len(ion) - 1]]
                attribute[189] = dicHy[peptide[-len(ion)]]
                attribute[190] = dicHy[peptide[-len(ion) - 1]]

                #b/y离子的碱性、螺旋性、疏水性
                attribute[191] = ion_b
                attribute[192] = pep_b - ion_b
                attribute[193] = ion_he
                attribute[194] = pep_he - ion_he
                attribute[195] = ion_hy
                attribute[196] = pep_hy - ion_hy

                #两个碎片离子的质量 
                attribute[203] = ion_mz
                attribute[204] = pep_mz - ion_mz
            else:
                #b/y离子的质量与肽质量的比
                attribute[182] = ion_mz / pep_mz
                attribute[181] = 1 - attribute[182]
                
                #碎裂点距肽C/N端的距离
                attribute[183] = len(ion)
                attribute[184] = len(peptide) - len(ion)

                #碎裂点两个氨基酸的碱性、螺旋性、疏水性
                attribute[185] = dicB[peptide[len(ion)]]
                attribute[186] = dicB[peptide[len(ion) - 1]]
                attribute[187] = dicHe[peptide[len(ion)]]
                attribute[188] = dicHe[peptide[len(ion) - 1]]
                attribute[189] = dicHy[peptide[len(ion)]]
                attribute[190] = dicHy[peptide[len(ion) - 1]]

                #b/y离子的碱性、螺旋性、疏水性
                attribute[192] = ion_b
                attribute[191] = pep_b - ion_b
                attribute[194] = ion_he
                attribute[193] = pep_he - ion_he
                attribute[196] = ion_hy
                attribute[195] = pep_hy - ion_hy

                #两个碎片离子的质量
                attribute[204] = ion_mz
                attribute[203] = pep_mz - ion_mz

            #碎裂点两个氨基酸的碱性、螺旋性、疏水性的差值和均值
            attribute[197] = (attribute[185] + attribute[186]) / 2.0
            attribute[198] = abs(attribute[185] - attribute[186])
            attribute[199] = (attribute[187] + attribute[188]) / 2.0
            attribute[200] = abs(attribute[187] - attribute[188])
            attribute[201] = (attribute[189] + attribute[190]) / 2.0
            attribute[202] = abs(attribute[189] - attribute[190])

            #Y离子的质荷比减去肽的质荷比
            attribute[205] = pep_mz - (ion_mz)/2.0
            
            #肽/两个碎片中碱性氨基酸的个数
            attribute[206] = peptide.count('K') + peptide.count("R") + peptide.count('H')
            attribute[207] = ion.count('K') + ion.count('R') + ion.count('H')
            attribute[208] = attribute[206] - attribute[207]

            #肽序列的长度
            attribute[209] = len(peptide)

            #肽的碱性、疏水性、螺旋性
            attribute[210] = pep_b
            attribute[211] = pep_he
            attribute[212] = pep_hy

            #肽的质荷比
            attribute[213] = pep_mz / 2.0
            cunt += 1
            flag=True
            #if cunt==24674:
             #   break;
        except:
            print('catch error in' + str(cunt)+' line')
        if flag:
            data=attribute.tolist()
            data.insert(0,cunt)
            data.insert(1,f_line[0])
            data.append(float(f_line[6].split('\n')[0]))
            writer.writerow(data)
            #print('write '+str(cunt)+' line in file')
    f.close()
wf.close()

##最大最小值归一化
#def max_min(X):
#    max_x = max(X)
#    min_x = min(X)
#    return [(x - min_x)/float(max_x - min_x) for x in X]

##方差均值归一化
#def z_score(X):
#    mean_x = numpy.mean(X)
#    s = sum((x - mean_x)*(x-mean_x) for x in X)/len(X)
#    return [(x - mean_x)/s for x in X]

#data = numpy.array(data)
#data = data.T
#for i in range(60):
#    data[i] = max_min(data[i])
#for i in range(183,214):
#    data[i] = max_min(data[i])
#data = data.T

#Pickle.dump((array(data[:40207]),labels[:40207]),f1,True)
#Pickle.dump((array(data[40207:80188]),labels[40207:80188]),f1,True)
#Pickle.dump((array(data[80188:]),labels[80188:]),f1,True)
#f1.close()

#f = open('test_spectra0-1.txt','w')
#for i in range(40207):
#    for j in range(214):
#        f.write(str(data[i][j])+'\t')
#    f.write(str(labels[i])+'\n')
#f.close()

#print "VICTORY!"

