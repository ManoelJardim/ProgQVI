import numpy as np
import pandas as pd
import sys
import time
import scipy as sc
from gams import *
from scipy.io import loadmat
from cvxopt import matrix, solvers

ws=GamsWorkspace()
r=100
n=5
al=0
l=np.array([[-0.1202],[-1.7418],[-2.7064],[-2.0502],[-4.4616 ]])
u=-l
A=np.array([[19.8699, 0.5369, 2.9482, 0.3358, 7.1239],[4.1819, 16.3484, -5.2030, 5.4332, 2.7143],[-5.6554, 0.9422, 19.0981, 7.1556, -7.3810],[-1.8770, 0.1918, -5.3596, 18.3565, -7.8847],[-6.0303, -3.6171, -1.4658, 4.6238, 15.4085]])
Adf=pd.DataFrame(A)
b=10*np.ones((5,1))
bvip=b
M=np.array([[1, 0, 0, 0, 0],[1, 1, 0, 0, 0],[1, 1, 1, 0, 0],[1, 1, 1, 1, 0],[1, 1, 1, 1, 1]])
c=np.array([[0.3070],[1.1186],[2.6149]])
cvip=c
Q1=al*M+np.array([[1.9073, 0.2403, 0.2352, -0.4903, -0.2651],[0.2403, 1.1319, 1.2087, -0.3268, 0.2540],[0.2352, 1.2087, 1.6862, 0.2941, 0.6732],[-0.4903, -0.3268, 0.2941, 1.8258, 0.1363],[-0.2651, 0.2540, 0.6732, 0.1363, 1.5527]])
Q1df=pd.DataFrame(Q1)
Q2=al*M+np.array([[2.7307, 0.5988, 1.5728, 1.4072, -0.3082],[0.5988, 2.2435, 0.7546, 1.3632, 1.5852],[1.5728, 0.7546, 2.3809, 1.2625, 1.0403],[1.4072, 1.3632, 1.2625, 1.7612, 0.3071],[-0.3082, 1.5852, 1.0403, 0.3071, 2.6305]])
Q2df=pd.DataFrame(Q2)
Q3=al*M+np.array([[2.5189, 2.1947, 1.7697, 2.2753, 1.9885],[2.1947, 3.8143, 1.3839, 1.5636, 1.8451],[1.7697, 1.3839, 3.3655, 1.6441, 1.9946],[2.2753, 1.5636, 1.6441, 3.6885, 2.3272],[1.9885, 1.8451, 1.9946, 2.3272, 2.2883]])
Q3df=pd.DataFrame(Q3)
output_file = "output_QVI.gms"
header_text = """\
set i / 0*"""+str(n-1)+""" /;
alias(i,j);
\n
"""
error=[]
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
    file.write("table Q1(i,j)\n")
    file.write('$onDelim\n')
    col_labels = list(map(str, Q1df.columns))
    file.write(delim + delim.join(col_labels) + '\n')
    for row_label, row in zip(Q1df.index, Q1df.itertuples(index=False)):
        file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
    file.write('$offDelim')
    file.write("\n;\n")
    file.write("table Q2(i,j)\n")
    file.write('$onDelim\n')
    col_labels = list(map(str, Q2df.columns))
    file.write(delim + delim.join(col_labels) + '\n')
    for row_label, row in zip(Q2df.index, Q2df.itertuples(index=False)):
        file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
    file.write('$offDelim')
    file.write("\n;\n")
    file.write("table Q3(i,j)\n")
    file.write('$onDelim\n')
    col_labels = list(map(str, Q3df.columns))
    file.write(delim + delim.join(col_labels) + '\n')
    for row_label, row in zip(Q3df.index, Q3df.itertuples(index=False)):
        file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
    file.write('$offDelim')
    file.write("\n;\n")
    file.write("parameters\n")
    file.write("b(i)/")
    num_rows_additional = bvip.shape[0]
    values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/,\n")
    file.write("u(i)/")
    num_rows_additional = u.shape[0]
    values_additional = [f"{row} {u[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/,\n")
    file.write("l(i)/")
    num_rows_additional = l.shape[0]
    values_additional = [f"{row} {l[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/,\n")
    file.write("c(i)/")
    num_rows_additional = cvip.shape[0]
    values_additional = [f"{row} {cvip[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/;\n\n")
    texto="""\
variables x(j),y(j);
x.up(j)=u(j);
x.lo(j)=l(j);
y.up(j)=u(j);
y.lo(j)=l(j);

equations
F(i),
g1,g2,g3;
F(i)..  sum{j, A(i,j)*x(j)} + b(i) =N= 0;
g1.. sum{i,x(i)*sum{j,Q1(i,j)*y(j)}}-c("0")=L=0;
g2.. sum{i,x(i)*sum{j,Q2(i,j)*y(j)}}-c("1")=L=0;
g3.. sum{i,x(i)*sum{j,Q3(i,j)*y(j)}}-c("2")=L=0;
model teste /all/;
file annotations / '%emp.info%' /;
putclose annotations  'qvi  F x y  g1 g2 g3'  ;
solve teste using emp;
    """
    file.write(texto)

job0=ws.add_job_from_file("output_QVI.gms")
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

##############AGORA PROG
#r=100 converge
yk=np.zeros((n,1))
xk=yk
# Nome do arquivo de saída
output_file = "output_VI.gms"
Tvi=0
erro=10
tol=0.001
ite=0
erro=1
while erro>tol:
    ite+=1
    Avip=A+r*np.eye(n)
    bvip=b-yk-r*xk
    xaux=xk-(1/r)*yk
    d=xaux
    cvip=c
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
        file.write("table Q1(i,j)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Q1df.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Q1df.index, Q1df.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("table Q2(i,j)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Q2df.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Q2df.index, Q2df.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("table Q3(i,j)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Q3df.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Q3df.index, Q3df.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("parameters\n")
        file.write("b(i)/")
        num_rows_additional = bvip.shape[0]
        values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("u(i)/")
        num_rows_additional = u.shape[0]
        values_additional = [f"{row} {u[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("l(i)/")
        num_rows_additional = l.shape[0]
        values_additional = [f"{row} {l[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("d(i)/")
        num_rows_additional = d.shape[0]
        values_additional = [f"{row} {d[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("c(i)/")
        num_rows_additional = cvip.shape[0]
        values_additional = [f"{row} {cvip[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/;\n\n")
        texto="""\
    variables y(j);
    y.up(j)=u(j);
    y.lo(j)=l(j);

    equations
    F(i),
    g1,g2,g3;
    F(i)..  sum{j, A(i,j)*y(j)} + b(i) =N= 0;
    g1.. sum{i,d(i)*sum{j,Q1(i,j)*y(j)}}-c("0")=L=0;
    g2.. sum{i,d(i)*sum{j,Q2(i,j)*y(j)}}-c("1")=L=0;
    g3.. sum{i,d(i)*sum{j,Q3(i,j)*y(j)}}-c("2")=L=0;
    model teste /all/;
    file annotations / '%emp.info%' /;
    putclose annotations  'vi  F y  g1 g2 g3'  ;
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
    erro=np.linalg.norm(x-xk)+np.linalg.norm(y-yk)
    xk=x
    yk=y
    print(erro)
print('TQVI',Tqvi, 'TVIs',Tvi)
print(ite)
