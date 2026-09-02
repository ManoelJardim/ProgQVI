'''
Opções:'MovSet3A1','MovSet3A2','MovSet3B1','MovSet3B2','MovSet4A1','MovSet4A2','MovSet4B1','MovSet4B2'
          'Box2A','Box2B','Box3A','Box3B'
          'RHS1A1','RHS1B1','RHS2A1','RHS2B1'
'''

model='RHS1B1'

r=0.001
Tqvi=0
import numpy as np
import pandas as pd
import sys
import time
import scipy as sc
from gams import *
from scipy.io import loadmat
from cvxopt import matrix, solvers

ws=GamsWorkspace()
def generateA(n):
    lambda_A = np.zeros((n,1))
    for i in range(n):
        lambda_A[i] = np.sin(((i-1)/n)*np.pi/2)*10+1
    
    vec_A = np.zeros((n,1))
    for i in range(n):
        vec_A[i] = 9*np.cos(((i-1)/n)*2*np.pi)+1
    
    V_A = np.zeros((n,n))
    for i in range(n):
        V_A[:,i] = vec_A.flatten()
    
    for i in range(n):
        c = 1
        for j in range(i,n):
            V_A[i,j] = V_A[i,j] + c
            c = c + 1
    
    Q, R = np.linalg.qr(V_A)
    A=Q@np.diag(lambda_A.flatten())@np.transpose(Q)
    eigA=np.linalg.eig(A)[0]
    return A, eigA



def generateM(n, a, eigA):
    vec_M = np.zeros((n,1))
    for i in range(n):
        vec_M[i] = 9*np.cos(((i-1)/n)*np.pi)+1

    M = np.zeros((n,n))
    for i in range(n):
        M[:,i] = vec_M.flatten() + (i-1)
    
    M = M*a*min(eigA)/(max(eigA)*np.linalg.norm(M,2))
    
    return M

error=[]

def generateQ(n):
    vec_Q = np.zeros((n,1))
    for i in range(n):
        vec_Q[i] = np.cos(((i-1)/n)*np.pi/2) + 0.1
    
    Q = np.diag(vec_Q.flatten())
    
    return Q


def generateA2(n):
    lambda_A = np.zeros((n,1))
    for i in range(n):
        lambda_A[i] = (-np.sin(((i-1)/n)*2*np.pi) + np.exp((i-1)/n)) * 10
    
    vec_A = np.zeros((n,1))
    for i in range(n):
        vec_A[i] = 9*np.cos(((i-1)/n)*2*np.pi) + 1
    
    V_A = np.zeros((n,n))
    for i in range(n):
        V_A[:,i] = vec_A.flatten()
    
    for i in range(n):
        c = 1
        for j in range(i,n):
            V_A[i,j] = V_A[i,j] + c
            c = c + 1
    
    A = V_A @ np.diag(lambda_A.flatten()) @ np.linalg.inv(V_A)
    
    return A


def generatea(n, alpha):
    a = np.zeros((n,1))
    for i in range(n):
        a[i] = (-np.cos(((i-1)/n)*20*np.pi) + np.exp((i-1)/n)) * 5
    
    max_a = np.max(a)
    a = a - max_a + 1 + alpha
    
    return a



def generatec(n, alpha):
    c = np.zeros((n,1))
    for i in range(n):
        c[i] = (-np.sin(((i-1)/n)*2*np.pi) + np.exp((i-1)/n)) * 5
    
    max_c = np.max(c)
    c = c - max_c + 1 + alpha
    
    return c



def generateM2(n):
    M = np.zeros((n,n))
    n2 = int(np.floor(n/2))
    
    for i in range(n2):
        for j in range(n2):
            M[i,j+n2] = np.sin(2*i+j)
    
    for i in range(n2):
        for j in range(n2):
            M[i+n2,j] = np.sin(i+2*j)
    
    return M



def generateE(n, m):
    vec_E = np.zeros((n,1))
    for i in range(n):
        vec_E[i] = np.cos(((i-1)/n)*np.pi) + 1
    
    E = np.zeros((m,n))
    for i in range(m):
        E[i,:] = (i-1) * vec_E.flatten()
    
    E[0:m,0:m] = E[0:m,0:m] + n * np.eye(m)
    
    return E


def generateC(n, m, A, E):
    C = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            C[i,j] = 3 * np.cos(i * np.pi / 10) + j / n
    
    #A = A.toarray()
    #E = E.toarray()
    #eigEAE = np.min(np.linalg.eig(np.transpose(E)@((A+np.transpose(A))@E)/2)[0])
    #p=eigEAE/(np.linalg.norm(A,2)*np.linalg.norm(E,2))
    #p=p/np.linalg.norm(C,2)
    return C


def generateC2(n, m):
    C = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            C[i,j] = 3 * np.cos(i * np.pi / 10) + j / n
    
    return C

if  model=='MovSet3A1':
    [QVItestA,eigA] = generateA(1000)
    QVItestM = generateM(1000,.1,eigA)
    QVItestQ = generateQ(1000)
    #np.savez('MovSet3A1.npz', QVItestA=QVItestA, QVItestM=QVItestM, QVItestQ=QVItestQ)



if  model=='MovSet3A2':
    [QVItestA,eigA] = generateA(2000)
    QVItestM = generateM(2000,.1,eigA)
    QVItestQ = generateQ(2000)
    #np.savez('MovSet3A2.npz', QVItestA=QVItestA, QVItestM=QVItestM, QVItestQ=QVItestQ)


if  model=='MovSet3B1':
    [QVItestA,eigA] = generateA(1000)
    QVItestM = generateM(1000,2,eigA)
    QVItestQ = generateQ(1000)
    #np.savez('MovSet3B1.npz', QVItestA=QVItestA, QVItestM=QVItestM, QVItestQ=QVItestQ)


if  model=='MovSet3B2':
    [QVItestA,eigA] = generateA(2000)
    QVItestM = generateM(2000,2,eigA)
    QVItestQ = generateQ(2000)
    #np.savez('MovSet3B2.npz', QVItestA=QVItestA, QVItestM=QVItestM, QVItestQ=QVItestQ)


if  model=='MovSet4A1':
    [QVItestA,eigA] = generateA(400)
    QVItestM = generateM(400,.1,eigA)
    #p.savez('MovSet4A1.npz', QVItestA=QVItestA, QVItestM=QVItestM)


if  model=='MovSet4A2':
    [QVItestA,eigA] = generateA(800)
    QVItestM = generateM(800,.1,eigA)
    #np.savez('MovSet4A2.npz', QVItestA=QVItestA, QVItestM=QVItestM)


if  model=='MovSet4B1':
    [QVItestA,eigA] = generateA(400)
    QVItestM = generateM(400,2,eigA)
    #np.savez('MovSet4B1.npz', QVItestA=QVItestA, QVItestM=QVItestM)



if  model=='MovSet4B2':
    [QVItestA,eigA] = generateA(800)
    QVItestM = generateM(800,2,eigA)
    #np.savez('MovSet4B2.npz', QVItestA=QVItestA, QVItestM=QVItestM)


if  model=='Box2A':
    QVItestA = generateA2(500)
    QVItesta = generatea(500,-1)
    QVItestc = generatec(500,-1)
    #np.savez('Box2A.npz', QVItestA=QVItestA, QVItesta=QVItesta, QVItestc=QVItestc)


if  model=='Box2B':
    QVItestA = generateA2(500)
    QVItesta = generatea(500,1)
    QVItestc = generatec(500,1)
    #np.savez('Box2B.npz', QVItestA=QVItestA, QVItesta=QVItesta, QVItestc=QVItestc)


if  model=='Box3A':
    QVItestA = generateA2(500)
    QVItestM = generateM2(500)
    QVItesta = generatea(500,-1)
    QVItestc = generatec(500,-1)
    #np.savez('Box3A.npz', QVItestA=QVItestA, QVItestM=QVItestM, QVItesta=QVItesta, QVItestc=QVItestc)


if  model=='Box3B':
    QVItestA = generateA2(500)
    QVItestM = generateM2(500)
    QVItesta = generatea(500,1)
    QVItestc = generatec(500,1)
    #np.savez('Box3B.npz', QVItestA=QVItestA, QVItestM=QVItestM, QVItesta=QVItesta, QVItestc=QVItestc)


if  model=='RHS1A1':
    QVItestA = generateA(200)
    #print(np.shape(QVItestA))
    QVItestE = generateE(200,199)
    #print(np.shape(QVItestE))
    QVItestC = generateC(200,199,QVItestA,QVItestE)
    #print(np.shape(QVItestC))
    #np.savez('RHS1A1.npz', QVItestA=QVItestA, QVItestE=QVItestE, QVItestC=QVItestC)


if  model=='RHS1B1':
    QVItestA = generateA(200)
    QVItestE = generateE(200,199)
    QVItestC = generateC2(200,199)
    #np.savez('RHS1B1.npz', QVItestA=QVItestA, QVItestE=QVItestE, QVItestC=QVItestC)


if  model=='RHS2A1':
    QVItestA = generateA(200)
    QVItestE = generateE(200,199)
    QVItestC = generateC(200,199,QVItestA,QVItestE)
    #np.savez('RHS2A1.npz', QVItestA=QVItestA, QVItestE=QVItestE, QVItestC=QVItestC)


if  model=='RHS2B1':
    QVItestA = generateA(200)
    QVItestE = generateE(200,199)
    QVItestC = generateC2(200,199)
    #np.savez('RHS2B1.npz', QVItestA=QVItestA, QVItestE=QVItestE, QVItestC=QVItestC)


A=QVItestA[0]
Adf=pd.DataFrame(QVItestA[0])
E=QVItestE
Edf=pd.DataFrame(QVItestE)
C=QVItestC
Cdf=pd.DataFrame(QVItestC)
[m,n]=np.shape(QVItestE)
b=10*np.ones((n,1))
d=10*np.ones((m,1))
bvip=b
dvip=d
output_file = "output_QVI.gms"
header_text = """\
set i / 0*"""+str(m-1)+""" /;
set j / 0*"""+str(n-1)+""" /;
alias(j,k);
\n
"""
delim=','
# Abrir o arquivo de saída em modo de escrita
with open(output_file, "w") as file:
    # Escrever a primeira tabela
    file.write(header_text)
    file.write("table A(j,k)\n")
    file.write('$onDelim\n')
    col_labels = list(map(str, Adf.columns))
    file.write(delim + delim.join(col_labels) + '\n')
    for row_label, row in zip(Adf.index, Adf.itertuples(index=False)):
        file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
    file.write('$offDelim')
    file.write("\n;\n")
    file.write("table C(i,j)\n")
    file.write('$onDelim\n')
    col_labels = list(map(str, Cdf.columns))
    file.write(delim + delim.join(col_labels) + '\n')
    for row_label, row in zip(Cdf.index, Cdf.itertuples(index=False)):
        file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
    file.write('$offDelim')
    file.write("\n;\n")
    file.write("table E(i,j)\n")
    file.write('$onDelim\n')
    col_labels = list(map(str, Edf.columns))
    file.write(delim + delim.join(col_labels) + '\n')
    for row_label, row in zip(Edf.index, Edf.itertuples(index=False)):
        file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
    file.write('$offDelim')
    file.write("\n;\n")
    file.write("parameters\n")
    file.write("d(i)/")
    num_rows_additional = dvip.shape[0]
    values_additional = [f"{row} {dvip[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/,\n")
    file.write("b(j)/")
    num_rows_additional = bvip.shape[0]
    values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/;\n\n")
    texto="""\
variables x(j),y(j);
equations
F(j),
g(i);
F(j)..  sum{k, A(j,k)*x(k)} + b(j) =N= 0;
g(i).. sum{j,E(i,j)*y(j)}-d(i)+sum{j,C(i,j)*sin(x(j))}=L= 0;
model teste /all/;
file annotations / '%emp.info%' /;
putclose annotations  'qvi  F x y  g'  ;
solve teste using emp;
    """
    file.write(texto)
'''
job0=ws.add_job_from_file("/home/mjardim/documentos/compete2/output_QVI.gms")
#job0=ws.add_job_from_file("C:\\Users\\B2AY\\Compete\\output_QVI.gms")
print('comecou QVI')
start=time.time()
job0.run()
end=time.time()
Tqvi=end-start
print('terminou QVI')
v = job0.out_db["y"]
x=[]
for rec in v:
    x.append(rec.level)
xd=np.array(x).reshape(n,1)
'''

##############AGORA PROG
yk=np.zeros((n,1))
xk=yk
# Nome do arquivo de saída
output_file = "output_VI_rh.gms"
Tvi=0
erro=10
tol=0.001
ite=0
while erro>tol:
    ite+=1
    Avip=A+r*np.eye(n)
    bvip=b-yk-r*xk
    xaux=xk-(1/r)*yk
    l=xaux
    lvip=l
    Adf=pd.DataFrame(Avip)
    header_text = """\
set i / 0*"""+str(m-1)+""" /;
set j / 0*"""+str(n-1)+""" /;
alias(j,k);
\n
    """
    delim=','
    # Abrir o arquivo de saída em modo de escrita
    with open(output_file, "w") as file:
        # Escrever a primeira tabela
        file.write(header_text)
        file.write("table A(j,k)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Adf.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Adf.index, Adf.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("table C(i,j)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Cdf.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Cdf.index, Cdf.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("table E(i,j)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Edf.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Edf.index, Edf.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("parameters\n")
        file.write("d(i)/")
        num_rows_additional = dvip.shape[0]
        values_additional = [f"{row} {dvip[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("l(j)/")
        num_rows_additional = l.shape[0]
        values_additional = [f"{row} {lvip[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("b(j)/")
        num_rows_additional = bvip.shape[0]
        values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/;\n\n")
        texto="""\
variables y(j);
equations
F(j),
g(i);
F(j)..  sum{k, A(j,k)*y(k)} + b(j) =N= 0;
g(i).. sum{j,E(i,j)*y(j)}-d(i)+sum{j,C(i,j)*sin(l(j))}=L= 0;
model teste /all/;
file annotations / '%emp.info%' /;
putclose annotations  'vi  F y  g'  ;
solve teste using emp;
"""
        file.write(texto)
    job1=ws.add_job_from_file()
    print('comecou VI')
    start=time.time()
    job1.run()
    end=time.time()
    Tvi=Tvi+end-start
    print('terminou VI')
    v = job1.out_db["y"]
    x=[]
    for rec in v:
        x.append(rec.level)
    xint=np.array(x).reshape(n,1)
    Tvi=Tvi+end-start
    x=0.5*(xint+xk-(1/r)*yk)
    y=-(r/2)*(xint-xk-(1/r)*yk)
    erro=np.linalg.norm(x-xk)+np.linalg.norm(y-yk)
    xk=x
    yk=y
    print(erro)
print('TQVI',Tqvi, 'TVIs',Tvi)
print(ite)
