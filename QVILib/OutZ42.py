import time
import numpy as np
import scipy as sc
import sys
import pandas as pd
from gams import *
from scipy.io import loadmat
from cvxopt import matrix, solvers



ws=GamsWorkspace()
n=4
A=np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
b=np.array([[1],[1],[1],[1]])
r=0.001
e=r/2
e=0
yk=np.zeros((n,1))
xk=yk
# Nome do arquivo de saída
output_file = "output_VI_out.gms"
error=[]
Tvi=0
erro=10
tol=0.001
ite=0
erron=1
erroauxa=1
while erro>tol:
    ite+=1
    Avip=A+r*np.eye(4)
    bvip=b-yk-r*xk
    cvip=xk-(1/r)*yk
    Adf=pd.DataFrame(Avip)
    header_text = """\
set i / 0*3 /;
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
variables y(j);
y.up(i)=0;

equations
F(i),
g(i);
F(i)..  sum{j, A(i,j)*y(j)} + b(i) =N= 0;
g(i).. -y(i)-2.5+c(i)+c(i)*c(i)=L= 0;

model m / F, g /;
file annotations / '%emp.info%' /;
putclose annotations  'vi  F y  g'  ;
solve m using emp;
        """
        file.write(texto)
    
    job0=ws.add_job_from_file()
    print('comecou VI')
    start=time.time()
    job0.run()
    end=time.time()
    print('terminou VI')
    v = job0.out_db["y"]
    x=[]
    for rec in v:
        x.append(rec.level)
    xint=np.array(x).reshape(4,1)
    Tvi=Tvi+end-start
    x=0.5*(xint+xk-(1/r)*yk)
    y=-(r/2)*(xint-xk-(1/r)*yk)
    erro=np.linalg.norm(x-xk)+np.linalg.norm(y-yk)
    print(erro)
    errox=np.linalg.norm(x-xk)
    erroy=np.linalg.norm(y-yk)
    xk=x
    yk=y
print(ite)
print(xk)
