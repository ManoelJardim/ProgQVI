import time
import numpy as np
import scipy as sc
import sys
import pandas as pd
from gams import *
from scipy.io import loadmat
from cvxopt import matrix, solvers

ws=GamsWorkspace()
n=2

A=np.array([[2, 8/3],[5/4, 2]])
B=np.array([[0, 1],[1, 0]])
Bdf=pd.DataFrame(B)
b=np.array([[-34],[-24.25]])
r=0.001
yk=np.zeros((n,1))
xk=np.zeros((n,1))
# Nome do arquivo de saída
output_file = "output_VI_out.gms"
Tvi=0
erro=10
tol=0.001
ite=0
erroauxa=1
error=[]
while erro>tol:
    ite+=1
    Avip=A+r*np.eye(2)
    bvip=b-yk-r*xk
    cvip=xk-(1/r)*yk
    Adf=pd.DataFrame(Avip)
    header_text = """\
set i / 0*"""+str(n-1)+""" /;
alias(i,j);
\n
"""
    delim=','
    # Abrir o arquivo de saída em modo de escrita
    with open(output_file, "w") as file:
        # Escrever a primeira tabela
        file.write(header_text)
        file.write("table A(i,j)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Adf.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Adf.index, Adf.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("table E(i,j)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Bdf.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Bdf.index, Bdf.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("parameters\nc(i)/")
        num_rows_additional = cvip.shape[0]
        values_additional = [f"{row} {cvip[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\nb(i)/")
        num_rows_additional = bvip.shape[0]
        values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/;\n\n")
        texto="""\
positive variables y(j);
y.up(i)=11;

equations
F(i),
g(i);
F(i)..  sum{j, A(i,j)*y(j)} + b(i) =N= 0;
g(i).. y(i)+sum{j,E(i,j)*c(j)}-15=L= 0;

model m / F, g /;
file annotations / '%emp.info%' /;
putclose annotations  'vi  F y  g'  ;
solve m using emp;
        """
        file.write(texto)
    
    job0=ws.add_job_from_file("output_VI_out.gms")
    print('comecou VI')
    start=time.time()
    job0.run()
    end=time.time()
    print('terminou VI')
    v = job0.out_db["y"]
    x=[]
    for rec in v:
        x.append(rec.level)
    xint=np.array(x).reshape(n,1)
    Tvi=Tvi+end-start
    x=0.5*(xint+xk-(1/r)*yk)
    y=-(r/2)*(xint-xk-(1/r)*yk)
    erro=np.linalg.norm(x-xk)+(1/r)*np.linalg.norm(y-yk)
    errorel=np.linalg.norm(yk)#np.linalg.norm(x-xk)/np.linalg.norm(xk)
    xk=x
    yk=y
    print(error) 
print(xk)
print(ite)
print(r)
