import time
import numpy as np
import scipy as sc
import sys
import pandas as pd
from gams import *
from scipy.io import loadmat
from scipy.optimize import linprog
from cvxopt import matrix, solvers

ws=GamsWorkspace(system_directory=" ")

r=0.001 
error=[]

######################################### Example 1
G=3
C=2
A=np.array([[1/3,1/3,1/3],[1/3,1/3,1/3]])
b=np.array([[1/2],[1/2]])
E=np.array([[1,1,1],[1,1,1]])

'''
######################################### Example 2
G=10
C=5
A=np.array([[1.0, 1.0, 3.0, 0.1, 0.1, 1.2, 2.0, 1.0, 1.0, 0.07],
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
[9.9, 0.1, 5.0, 0.2, 6.0, 0.2, 8.0, 1.0, 1.0, 0.2],
[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
[1.0, 13.0, 11.0, 9.0, 4.0, 0.9, 8.0, 1.0, 2.0, 10.0]])

E=np.array([[0.6, 0.2, 0.2, 20.0, 0.1, 2.0, 9.0, 5.0, 5.0, 15.0],
[0.2, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 5.0, 5.0, 9.0],
[0.4, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 5.0, 7.0, 12.0],
[1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0, 3.0, 17.0],
[8.0, 1.0, 22.0, 10.0, 0.3, 0.9, 5.1, 0.1, 6.2, 11.0]])

b=np.array([[2],[1.3],[3],[0.2],[0.6]])

########################################## Example 3
G=50
C=10
A=(1/G)*np.ones((C,G))
b=(1/C)*np.ones((C,1))
E=np.ones((C,G))

#########################################
'''

n=G*(C+1)
Adf=pd.DataFrame(A)
Edf=pd.DataFrame(E) 

######################################### SOLVING DIRECTLY USING EMP
# Nome do arquivo de saída
output_file = "output_table_direto_Wal.gms"

# Texto a ser adicionado no início do arquivo
header_text = """\
set i / 0*"""+str(C-1)+""" /;
set j / 0*"""+str(G-1)+""" /;
Alias (i,k);
Alias (j,l);\n
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
    col_labels = list(map(str, Edf.columns))
    file.write(delim + delim.join(col_labels) + '\n')
    for row_label, row in zip(Edf.index, Edf.itertuples(index=False)):
        file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
    file.write('$offDelim')
    file.write("\n;\n\n")
    file.write("parameters\n")
    file.write("b(i)/")
    num_rows_additional = b.shape[0]
    values_additional = [f"{row} {b[row, 0]}" for row in range(num_rows_additional)]
    file.write(", ".join(values_additional))
    file.write("/;\n")
    texto="""\
variables iso_obj , agent_obj(i);

positive variables p(j) , x(i,j);
x.lo(i,j)=0.001;

equations iso_defobj , agent_defobj(i), preco, rest(i);

iso_defobj.. iso_obj =E= sum(j,p(j)*(sum(i,x(i,j)-E(i,j))));

agent_defobj(i).. agent_obj(i)=E=sum(j,(a(i,j)**(1/b(i)))*(x(i,j)**((b(i)-1)/b(i))))**(b(i)/(b(i)-1));

preco.. sum(j,p(j))=E=1;

rest(i).. sum(j,p(j)*(x(i,j)-E(i,j)))=L=0;

model m_oligop / iso_defobj, agent_defobj, preco, rest/;
file empinfo / '%emp.info%' /;
put empinfo 'equilibrium' /;
put 'max', iso_obj ,
loop (j, put p(j););
put iso_defobj , preco /;
loop (i,
      put 'max', agent_obj(i);
      loop (j, put x(i,j););
      put  agent_defobj(i), rest(i) /; ) ;
putclose empinfo;
solve m_oligop using emp;
"""
    file.write(texto)
    
job0=ws.add_job_from_file("output_table_direto_Wal.gms")
print('comecou QVI')
start=time.time()
job0.run()
end=time.time()
print('terminou QVI')
v = job0.out_db["x"]
x=[]
for rec in v:
    x.append(rec.level)
x=np.array(x).reshape(n-G,1)
v = job0.out_db["p"]
pr=[]
for rec in v:
    pr.append(rec.level)
pr=np.array(pr).reshape(G,1)
x=np.vstack((pr,x))

Tqvi=end-start
print('\n Tempo(s)= ',Tqvi)
xd=x

############################################## PROGQVI
n=(C+1)*G
yk=np.zeros((n,1))
xk=yk
# Nome do arquivo de saída
output_file = "output_VI.gms"
Tvi=0
erro=1
tol=0.001
ite=0
A=np.vstack((0.0000001*np.ones((1,G)),A))
Adf=pd.DataFrame(A)
E=np.vstack((0.00000001*np.ones((1,G)),E))
Edf=pd.DataFrame(E)
b=np.vstack((.5,b))
o=np.vstack((1,np.zeros((C,1))))
q=np.vstack((0,np.ones((C,1))))
s=np.vstack((1,np.zeros((C,1))))
while erro>tol:
    ite+=1
    d=-yk-r*xk
    D=d.reshape(C+1,G)
    Ddf=pd.DataFrame(D)
    xaux=xk-(1/r)*yk
    pr=xaux[0:G]
    #yr=xaux[G:2*G]
    header_text = """\
set i / 0*"""+str(C)+""" /;
set j / 0*"""+str(G-1)+""" /;
Alias (j,k);\n
Alias (i,l);\n
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
        file.write("table D(i,j)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, Ddf.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(Ddf.index, Ddf.itertuples(index=False)):
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
        file.write("b(i)/")
        num_rows_additional = b.shape[0]
        values_additional = [f"{row} {b[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("o(i)/")
        num_rows_additional = o.shape[0]
        values_additional = [f"{row} {o[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("q(i)/")
        num_rows_additional = q.shape[0]
        values_additional = [f"{row} {q[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/,\n")
        file.write("pr(j)/")
        num_rows_additional = pr.shape[0]
        values_additional = [f"{row} {pr[row, 0]}" for row in range(num_rows_additional)]
        file.write(", ".join(values_additional))
        file.write("/;\n")
        texto="""\
variables y(i,j);
y.lo(i,j)=0.00000001;
equations
F(i,j),
g(i),
g2;
F(i,j)..o(i)*(sum{l$(ord(l)>1),E(l,j)-y(l,j)})-q(i)*(A(i,j)**(1/b(i)))*((sum{k, (A(i,k)**(1/b(i)))*(y(i,k)**((b(i)-1)/b(i)))})**(1/(b(i)-1)))*(y(i,j)**(-1/(b(i))))+"""+str(r)+"""*y(i,j)+D(i,j) =N= 0;
g(i).. q(i)*sum{j,pr(j)*(y(i,j)-E(i,j))}=L=0;
g2.. sum{j,y('0',j)}=E=1;
model teste /all/;
file annotations / '%emp.info%' /;
putclose annotations  'vi  F y  g g2'  ;
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
    errorel=np.linalg.norm(yk)#np.linalg.norm(x-xk)/np.linalg.norm(xk)
    if ite>=3:
        error=np.append(error,errorel)
        if ite>=6 and error[-1]>error[-2]:
            r=1.1*r
        '''if ite>=6:
            error=error[-3:]
        if error[-1]==np.max(error):
            r=min(2*r,1000000)
            print(r)'''
    print(error)
    xk=x
    yk=y
    print(erro)
print('TQVI',Tqvi, 'TVIs',Tvi)
print(xk-xd)
print(ite)
print(r)
