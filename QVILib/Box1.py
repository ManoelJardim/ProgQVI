import numpy as np
import pandas as pd
import sys
import time
import scipy as sc
from gams import *
from scipy.io import loadmat
from cvxopt import matrix, solvers

ws=GamsWorkspace()

n=5
A=np.array([[19.8699, 0.5369, 2.9482, 0.3358, 7.1239],
    [4.1819, 16.3484, -5.2030, 5.4332, 2.7143],
    [-5.6554, 0.9422, 19.0981, 7.1556, -7.3810],
    [-1.8770, 0.1918, -5.3596, 18.3565, -7.8847],
    [-6.0303, -3.6171, -1.4658, 4.6238, 15.4085]])


Adf=pd.DataFrame(A)

b=np.array([[10],[10],[10],[10],[10]])
al=0.1
r=0.001
e=0
bvip=b
c=np.array([[0.1202],[1.7418],[2.7064],[2.0502],[4.4616]])
cvip=c
error=[]
output_file = "output_QVI.gms"
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
    file.write("parameters\n")
    file.write("b(i)/")
    num_rows_additional = bvip.shape[0]
    values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/,\n")
    file.write("c(i)/")
    num_rows_additional = cvip.shape[0]
    values_additional = [f"{row} {cvip[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/;\n\n")
    texto="""\
variables x(j),y(j);
equations
F(i),
g1(i),g2(i);
F(i)..  sum{j, A(i,j)*x(j)} + b(i) =N= 0;
g1(i).. y(i)-"""+str(al)+"""*x(i)-c(i)=L= 0;
g2(i).. -y(i)+"""+str(al)+"""*x(i)-c(i) =L= 0;
model teste /all/;
file annotations / '%emp.info%' /;
putclose annotations  'qvi  F y x  g1 g2'  ;
solve teste using emp;
    """
    file.write(texto)
job0=ws.add_job_from_file("output_QVI.gms")
print('comecou QVI')
start=time.time()
job0.run()
end=time.time()
Tqvi=end-start
resv=[]
print('terminou QVI')
v = job0.out_db["y"]
x=[]
for rec in v:
    x.append(rec.level)
xd=np.array(x).reshape(n,1)
print(xd)
res=1
##############AGORA PROG
yk=np.zeros((n,1))
xk=yk
# Nome do arquivo de saída
output_file = "output_VI.gms"
Tvi=0
erro=10
tol=0.0001
ite=0
erron=1
while erro>tol:
    ite+=1
    Avip=A+r*np.eye(n)
    bvip=b-yk-r*xk
    xaux=xk-(1/r)*yk
    cvip1=-al*xaux-c
    cvip2=al*xaux-c
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
        file.write("parameters\n")
        file.write("\nb(i)/")
        num_rows_additional = bvip.shape[0]
        values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("\nc1(i)/")
        num_rows_additional = cvip1.shape[0]
        values_additional = [f"{row} {cvip1[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("c2(i)/")
        num_rows_additional = cvip2.shape[0]
        values_additional = [f"{row} {cvip2[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/;\n\n")
        texto="""\
variables y(j);

equations
F(i),
g1(i),g2(i);
F(i)..  sum{j, A(i,j)*y(j)} + b(i) =N= 0;
g1(i).. y(i)+c1(i)=L= 0;
g2(i).. -y(i)+c2(i)=L=0;

model teste / all /;
file annotations / '%emp.info%' /;
putclose annotations  'vi  F y  g1 g2'  ;
solve teste using emp;
"""
        file.write(texto)
    job1=ws.add_job_from_file("output_VI.gms")
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
    errox=np.linalg.norm(x-xk)
    erroy=np.linalg.norm(y-yk)
    erro=errox+erroy
    xk=x
    yk=y
    print(erro)
print('TQVI',Tqvi, 'TVIs',Tvi)
print(ite)
print(xd,x)
