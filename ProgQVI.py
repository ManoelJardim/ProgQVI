#EMP, DW, ProQVI
import time
import numpy as np
import scipy as sc
import sys
import pandas as pd
from gams import *
from scipy.io import loadmat
from cvxopt import matrix, solvers
Tdwcum=[0]
Tdiretocum=[0]
Tprcum=[0]
Totalit=[0]
Totalitprog=[0]
conta=1
lamx=1
lamy=1
ws=GamsWorkspace()
while conta<=20:
    C=10
    P=10
    n=P*(C+2)
    auxy=np.vstack((np.ones((P,1)),np.zeros((C*P,1))))
    A1=np.ndarray(shape=(C,P,P))
    A1[0]=-1+2*np.random.rand(P,P)
    A1[0]=np.transpose(A1[0])@A1[0]
    A1[0]=10*(A1[0]/np.max(A1[0]))
    Aaux=np.kron(np.eye(C)[0,:],A1[0])
    for i in range(1,C):
        A1[i]=-1+2*np.random.rand(P,P)
        A1[i]=np.transpose(A1[i])@A1[i]
        A1[i]=10*(A1[i]/np.max(A1[i]))
        Aaux=np.vstack((Aaux,np.kron(np.eye(C)[i,:],A1[i])))

    ep=10*np.random.rand(C*P,1)
    Mep=ep.reshape(C,P)
    FirmMax=100*C*P
    l1=np.zeros((P,P))
    for i in range(C):    
        l1=np.hstack((l1,-np.eye(P)))
        
    blocoaux=np.zeros((P*C,P))
    linaux=np.hstack((blocoaux,Aaux))
    A=np.vstack((l1,linaux))
    colauxx=np.vstack((np.eye(P),np.zeros((C*P,P))))
    A=np.hstack((A,colauxx))
    linauxx=np.hstack((-np.eye(P),np.zeros((P,P*(C+1)))))
    A=np.vstack((A,linauxx))
    b=np.sum(Mep,axis=0).reshape(-1,1)
    A=A.round(3)
    A1=A1.round(3)
    ep=ep.round(3)
    Mep=Mep.round(3)
    Mep1=Mep
    for i in range(1,C+1):
        baux=10*np.random.rand(P,1)
        b=np.vstack((b,-baux))

    b=b.round(3)
    b=np.vstack((b,np.zeros((P,1))))
     
    def F(y):
        F=A@y+b
        return F

    ep=np.vstack((np.zeros((P,1)),ep))
    Adf=pd.DataFrame(A)

    #########################################FERRIS
    Q=np.ndarray(shape=(C,P,P)) 
    Q[0]=A1[0]
    Qd=Q[0]
    for i in range(1,C):
        Q[i]=A1[i]
        Qd=np.vstack((Qd,Q[i]))
        
    indicesQ = []
    for i in range(C):
        for j in range(P):
            indicesQ.append(str(i) + '.' + str(j))
    indicesQ=np.transpose(np.array(indicesQ))
    Qdf=pd.DataFrame(Qd)
    Qdf.index=indicesQ    
    # Nome do arquivo de saída
    output_file = "output_table_direto_Wal.gms"
    Mb=np.transpose(-b[P:P*(C+1)]).reshape(C,P)
    epdf=pd.DataFrame(Mep)
    bdf=pd.DataFrame(Mb)
    # Texto a ser adicionado no início do arquivo
    header_text = """\
    set i / 0*"""+str(C-1)+""" /;
    set k / 0*"""+str(P-1)+""" /;
    Alias (r,k);
    Alias (s,k);
    Alias (i,j);\n
    """
    delim=','
    # Abrir o arquivo de saída em modo de escrita
    with open(output_file, "w") as file:
        # Escrever a primeira tabela
        file.write(header_text)
        file.write("table b(i,k)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, bdf.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(bdf.index, bdf.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n")
        file.write("table ep(i,k)\n")
        file.write('$onDelim\n')
        col_labels = list(map(str, epdf.columns))
        file.write(delim + delim.join(col_labels) + '\n')
        for row_label, row in zip(epdf.index, epdf.itertuples(index=False)):
            file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
        file.write('$offDelim')
        file.write("\n;\n\n")
        file.write("table Q(i,r,s)\n")
        Qdf=Qdf.to_string()
        file.write(Qdf)
        file.write('\n')
        file.write(";\n\n")
        texto="""\
    variables iso_obj , agent_obj(i), z, firma_obj;

    positive variables p(k) , x(i,k), y(k);

    equations iso_defobj , agent_defobj(i), firma_defobj, preco, rest(i), restF;

    iso_defobj.. iso_obj =E= sum(k,p(k)*(sum(i,x(i,k)-ep(i,k))-y(k)));

    firma_defobj.. firma_obj=E= sum(k,p(k)*y(k));

    agent_defobj(i).. agent_obj(i)=E=-0.5*(sum((r,s),x(i,r)*Q(i,r,s)*x(i,s)))+sum(k,b(i,k)*x(i,k));

    preco.. sum(k,p(k))=E=1;

    rest(i).. sum(k,p(k)*(x(i,k)-ep(i,k)))=L=0;

    restF.. sum{k,y(k)**2}=L="""+str(FirmMax)+""";

    model m_oligop / iso_defobj , firma_defobj, agent_defobj , preco , rest, restF /;

    file empinfo / '%emp.info%' /;

    put empinfo 'equilibrium' /;

    put 'max', iso_obj ,
    loop (k, put p(k););
    put iso_defobj , preco /;

    put 'max', firma_obj,
    loop (k, put y(k););
    put firma_defobj, restF/;

    loop (i,
          put 'max', agent_obj(i);
          loop (k, put x(i,k););
          put  agent_defobj(i), rest(i) /; ) ;

    putclose empinfo;

    solve m_oligop using emp;
        """
        file.write(texto)

    job=ws.add_job_from_file("output_table_direto_Wal.gms")
    print('comecou QVI')
    start=time.time()
    job.run()
    end=time.time()
    print('terminou QVI')
    v = job.out_db["x"]
    x=[]
    for rec in v:
        x.append(rec.level)
    x=np.array(x).reshape(n-2*P,1)
    v = job.out_db["p"]
    pr=[]
    for rec in v:
        pr.append(rec.level)
    pr=np.array(pr).reshape(P,1)
    v = job.out_db["y"]
    fi=[]
    for rec in v:
        fi.append(rec.level)
    fi=np.array(fi).reshape(P,1)
    x=np.vstack((pr,x,fi))
    print('\n x=',x)
    Tqvi=end-start
    print('\n Tempo(s)= ',Tqvi)
    xd=x

    ###################################################### Decompositions
    Mep=ep[P:2*P]
    for i in range(2,C+1):
        Mep=np.hstack((Mep,ep[P*i:P*(i+1)]))
        
    u=1000*np.ones((n-2*P,1))
    delim=','
    def VI_cvx(parc):
        bparc=(b+parc)[P:P*(C+1)]
        Q = matrix(A1[0])
        p = matrix(bparc[0:P])
        G = matrix(-np.eye(P))
        h = matrix(np.zeros((P,1)))
        solT=solvers.qp(Q, p, G, h)
        solT = np.array(solT['x'])
        for i in range(1,C):
            Q=matrix(A1[i])
            p = matrix(bparc[P*i:P*(i+1)])
            sol=solvers.qp(Q, p, G, h)
            sol = np.array(sol['x'])
            solT=np.vstack((solT,sol))
        return solT

    def VIGAMS_(parc):
        Avidf=pd.DataFrame(A[P:(C+1)*P,P:(C+1)*P])
        bvi=b[P:(C+1)*P]+parc[P:(C+1)*P]
        output_file = "output_table_VI.gms"
        header_text = """\
    set i / 0*"""+str(n-2*P-1)+""" /;
    alias (i,k);
    alias (i,l); \n
    """
        # Abrir o arquivo de saída em modo de escrita
        with open(output_file, "w") as file:
            file.write(header_text)
            file.write("table A(i,k)\n")
            file.write('$onDelim\n')
            col_labels = list(map(str, Avidf.columns))
            file.write(delim + delim.join(col_labels) + '\n')
            for row_label, row in zip(Avidf.index, Avidf.itertuples(index=False)):
                file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
            file.write('$offDelim')
            file.write("\n;\n\n")
       
            '''file.write("table Ma(i,l,k)\n")
            file.write(Madf)
            file.write('\n')
            file.write(";\n\n")'''
            
            # Adicionar parâmetro adicional a partir de um array
            file.write("parameters\nb(i)/")
            num_rows_additional = bvi.shape[0]
            values_additional = [f"{row} {bvi[row, 0]}" for row in range(num_rows_additional)]
            file.write(", ".join(values_additional))
            file.write("/;\n\n")
            file.write("parameters P;\n")
            file.write("P="+str(P)+";\n")
            # Adicionar parâmetro adicional a partir de um array
            
            # Escrever o texto adicional ao final do arquivo
            file.write("positive variables\n")
            file.write("  yvi(k)  'variable of interest, aka decision variable'\n")
            file.write("  ;\n")
            file.write("equations\n")
            file.write("  F(i)   'FOC for agent optimization models'\n")
            file.write("  h   \n")
            file.write(";\n")
            file.write("F(i)..  sum{k, A(i,k)*yvi(k)} + b(i) =N= 0;\n")
            file.write("model linVI / F /;\n")
            file.write("file annotations / '%emp.info%' /;\n")
            file.write("putclose annotations  'vi  F yvi'  ;\n")
            file.write("solve linVI using emp;")
        
        jobvi=ws.add_job_from_file("output_table_VI.gms")
        jobvi.run()
        vvi = jobvi.out_db["yvi"]
        yvi=[]
        for rec in vvi:
            yvi.append(rec.level)
        yvi=np.array(yvi).reshape(n-2*P,1)
        return yvi

    tole=-0.01
    k=2
    ts0=time.time()
    ts=time.time()-ts0
    Tdw=ts
    x0=0.00000001*np.ones((n,1))
    for i in range(P):
        x0[i,0]=1/P

    xs=(np.sqrt(FirmMax)/P)*np.ones((n,1))
    for i in range(P):
        xs[i,0]=0
    xs[2,0]=1
    Xs=np.hstack((x0,xs))
    Xs0=Xs[0:P,:]
    Ma=np.ndarray(shape=(C,k,k))
    for i in range(C):
        Xsi=Xs[P*(i+1):P*(i+2),:] 
        Ma[i]=np.transpose(Xs0)@Xsi

    GAP=-1
    tc=0
    p=ep
    raux=np.vstack((np.ones((P,1)),0*np.ones((P,1))))
    saux=np.vstack((0*np.ones((P,1)),np.ones((P,1))))
    while (GAP<tole)and(k<=500):
        Ait=np.transpose(Xs)@A@Xs
        bit=np.transpose(Xs)@b
        Adf=pd.DataFrame(Ait)        
        Qd=Ma[0]
        for i in range(1,C):
            Qd=np.vstack((Qd,Ma[i]))
      
        indicesQ = []
        for i in range(C):
            for j in range(k):
                indicesQ.append(str(i) + '.' + str(j))
        indicesQ=np.transpose(np.array(indicesQ))
        Qdf=pd.DataFrame(Qd)
        Qdf.index=indicesQ    
        Maux=np.transpose(Xs0)@Mep
        Mauxdf=pd.DataFrame(Maux)
        ########################################################
        output_file = "output_table.gms"
        header_text = """\
    set j / 0*"""+str(C-1)+""" /;
    set k / 0*"""+str(k-1)+""" /; 
    set u / 0*1 /; 
    alias(k,i);
    alias(k,l);\n
    """
        delim=','
        with open(output_file, "w") as file:
            file.write(header_text)
            file.write("table A(i,k)\n")
            file.write('$onDelim\n')
            col_labels = list(map(str, Adf.columns))
            file.write(delim + delim.join(col_labels) + '\n')
            for row_label, row in zip(Adf.index, Adf.itertuples(index=False)):
                file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
            file.write('$offDelim')
            file.write("\n;\n\n")
            file.write("table Maux(k,j)\n")
            file.write('$onDelim\n')
            col_labels = list(map(str, Mauxdf.columns))
            file.write(delim + delim.join(col_labels) + '\n')
            for row_label, row in zip(Mauxdf.index, Mauxdf.itertuples(index=False)):
                file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
            file.write('$offDelim')
            file.write("\n;\n\n")
            file.write("table Q(j,l,k)\n")
            Qdf=Qdf.to_string()
            file.write(Qdf)
            file.write('\n')
            file.write(";\n\n")
            num_rows = bit.shape[0]
            file.write("parameters\nb(i)/")
            values = [f"{row} {bit[row, 0]}" for row in range(num_rows)]
            file.write(", ".join(values))
            file.write("/;\n")
            file.write("positive variables\n")
            file.write("  ty(k)  'variable of interest, aka decision variable'\n")
            file.write("  tx(k)  'parameter variable shadowing y'\n")
            file.write("  ;\n")
            file.write("ty.up(k) = 1;  tx.up(k) = 1;\n")
            file.write("equations\n")
            file.write("  F(i)   'FOC for agent optimization models'\n")
            file.write("  g(j)   'define feasible set K(tx) for QVI'\n")
            file.write("  h   'define feasible set K(tx) for QVI'\n")
            file.write(";\n")
            file.write("F(i)..  sum{k, A(i,k)*ty(k)} + b(i) =N= 0;\n\n")
            file.write("g(j)..  sum{l,tx(l)*(sum{k, Q(j,l,k)*ty(k)})} - sum{k,tx(k)*Maux(k,j)} =L= 0;\n")
            file.write("h..  sum{l,ty(l)} =E= 1;\n")
            file.write("model m / F, h, g /;\n")
            file.write("file annotations / '%emp.info%' /;\n")
            file.write("putclose annotations  'qvi  F ty tx h g'  ;\n")
            file.write("solve m using emp;")
        
        job=ws.add_job_from_file("output_table.gms")
        tm0=time.time()
        job.run()
        tm=time.time()-tm0
        v = job.out_db["tx"]
        t=[]
        for rec in v:
            t.append(rec.level)
        t=np.array(t).reshape(k,1)
        mult = job.out_db["g"]
        lam=[]
        for rec in mult:
            lam.append(rec.marginal)
        lam=np.array(lam).reshape(C,1)
        xM=Xs@t  
        lam=-lam
        Jac=np.zeros((C,P*(C+1)))
        auxxM=np.squeeze(xM[0:P])
        for i in range(C):
            Jac[i,P*(i+1):P*(i+1)+P]=auxxM
        
        Jac=np.hstack((Jac,np.zeros((C,P))))
        parc=np.transpose(Jac)@lam
        ts0=time.time()
        xsred=VI_cvx(parc)
        ts=time.time()-ts0
        v1=F(xM)[0:P]-xM[P*(C+1):]
        bvip=np.vstack((v1,np.zeros((P,1))))+np.vstack((parc[0:P],parc[P*(C+1):]))
        Avip=np.vstack((np.hstack((np.zeros((P,P)),np.eye(P))),np.hstack((-np.eye(P),np.zeros((P,P))))))
        Avipdf=pd.DataFrame(Avip)
        output_file = "output_table_VIP.gms"
        header_text = """\
    set i / 0*"""+str(2*P-1)+""" /;
    alias (i,k);
    alias (i,l); \n
    """
        # Abrir o arquivo de saída em modo de escrita
        with open(output_file, "w") as file:
            file.write(header_text)
            file.write("table A(i,k)\n")
            file.write('$onDelim\n')
            col_labels = list(map(str, Avipdf.columns))
            file.write(delim + delim.join(col_labels) + '\n')
            for row_label, row in zip(Avipdf.index, Avipdf.itertuples(index=False)):
                file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
            file.write('$offDelim')
            file.write("\n;\n\n") 
            # Adicionar parâmetro adicional a partir de um array
            file.write("parameters\nb(i)/")
            num_rows_additional = bvip.shape[0]
            values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
            file.write(", ".join(values_additional))
            file.write("/\n")
            num_rows_additional = raux.shape[0]
            file.write("r(l)/")
            values_additional = [f"{row} {raux[row, 0]}" for row in range(num_rows_additional)]
            file.write(", ".join(values_additional))
            file.write("/\n")
            file.write("s(l)/")
            values_additional = [f"{row} {saux[row, 0]}" for row in range(num_rows_additional)]
            file.write(", ".join(values_additional))
            file.write("/;\n\n")
            # Escrever o texto adicional ao final do arquivo
            file.write("positive variables\n")
            file.write("  yvip(k)  'variable of interest, aka decision variable'\n")
            file.write("  ;\n")
            #file.write("yvi.up(k)=u(k);\n")
            file.write("equations\n")
            file.write("  F(i)   'FOC for agent optimization models'\n")
            file.write("  h   \n g \n")
            file.write(";\n")
            file.write("F(i)..  sum{k, A(i,k)*yvip(k)} + b(i) =N= 0;\n")
            file.write("h..  sum{l,r(l)*yvip(l)}=E= 1;\n")#Aqui troquei P-1 por P
            file.write("g..  sum{l,s(l)*yvip(l)**2}=L=" +str(FirmMax)+";\n")#Aqui troquei P-1 por P
            file.write("model VI / F, g, h /;\n")
            file.write("file annotations / '%emp.info%' /;\n")
            file.write("putclose annotations  'vi  F yvip g h'  ;\n")
            file.write("solve VI using emp;")
        
        jobvip=ws.add_job_from_file("output_table_VIP.gms")
        #print('comecou VI')
        ts0=time.time()
        jobvip.run()
        ts=ts+time.time()-ts0
        #print('terminou VI')
        vvip = jobvip.out_db["yvip"]
        yvip=[]
        for rec in vvip:
            yvip.append(rec.level)
        yvip=np.array(yvip).reshape(2*P,1)
        xs=np.vstack((yvip[0:P],xsred,yvip[P:]))
        Tdw=Tdw+tm+ts
        #print('terminou VI')
        Xs=np.hstack((Xs,xs))
        Xs0=Xs[0:P,:]
        k+=1 
        Ma=np.ndarray(shape=(C,k,k))
        for i in range(C):
            Xsi=Xs[P*(i+1):P*(i+2),:]
            Ma[i]=np.transpose(Xs0)@Xsi
        #print('Xs=',Xs,'\n')
        GAP=np.transpose(F(xM)+parc)@(xs-xM)
        print('GAP=',GAP[0,0])
        
    print('\n Iterações= ',k-1)
    print('Tempo DW: ',Tdw)    
    Tdwcum.append(Tdw)
    Totalit.append(k-1)
    print('Tempo GNEP_WAL direto: ',Tqvi)
    Tdiretocum.append(Tqvi)
    solvers.options['show_progress']=False
    def VI_cvx2(r,y,xk):
        pre=np.transpose(xk[0:P]-(1/r)*y[0:P])
        const=pre@ep[P:2*P]
        bparc=(b-y-r*xk)[P:P*(C+1)]
        Q = matrix(A1[0]+r*np.eye(P))
        p = matrix(bparc[0:P])
        G = matrix(np.vstack((-np.eye(P),pre)))
        h = matrix(np.vstack((np.zeros((P,1)),const)))
        solT=solvers.qp(Q, p, G, h)
        solT = np.array(solT['x'])
        for i in range(1,C):
            const=pre@ep[P*(i+1):P*(i+2)]
            Q=matrix(A1[i]+r*np.eye(P))
            p = matrix(bparc[P*i:P*(i+1)])
            h = matrix(np.vstack((np.zeros((P,1)),const)))
            sol=solvers.qp(Q, p, G, h)
            sol = np.array(sol['x'])
            solT=np.vstack((solT,sol))
        return solT

    r=0.001
    yk=np.zeros((n,1))
    xk=yk
    xk[0:P]=(1/P)*np.ones((P,1))
    tprog=0
    crit=1
    ite=1
    teta=0.5
    ep_sum=np.sum(Mep1,axis=0)
    ep_sum=np.transpose(ep_sum[np.newaxis,:])
    saux=np.vstack((np.ones((P,1)),0*np.ones((P,1))))
    raux=np.vstack((0*np.ones((P,1)),np.ones((P,1))))
    delim=','
    row_labels = [str(i) for i in range(P*C)]  # Rótulos de 0 até P-1
    col_labels = row_labels  # Rótulos de 0 até C-1
    while crit>=0.001 and ite<=300:
        xintvi=VI_cvx2(r,yk,xk)
        d=yk+r*xk
        d1=d[P*(C+1):]
        d2=d[0:P]
        Mxk=xintvi.reshape(C,P)
        xk_sum=np.sum(Mxk,axis=0)
        xk_sum=np.transpose(xk_sum[np.newaxis,:])
        bvip=np.vstack((-d1,ep_sum-xk_sum-d2))
        bvip=bvip.round(5)
        Avip=np.vstack((np.hstack((r*np.eye((P)),-np.eye((P)))),np.hstack((((np.eye((P)),r*np.eye((P))))))))
        Avipdf=pd.DataFrame(Avip)
        output_file = "output_table_VIP2.gms"
        header_text = """\
    set i / 0*"""+str(2*P-1)+""" /;
    alias (i,k);
    alias (i,l); \n
    """
        with open(output_file, "w") as file:
            file.write(header_text)
            file.write("table A(i,k)\n")
            file.write('$onDelim\n')
            col_labels = list(map(str, Avipdf.columns))
            file.write(delim + delim.join(col_labels) + '\n')
            for row_label, row in zip(Avipdf.index, Avipdf.itertuples(index=False)):
                file.write(str(row_label) + delim + delim.join(map(str, row)) + '\n')
            file.write('$offDelim')
            file.write("\n;\n\n")            
            # Adicionar parâmetro adicional a partir de um array
            file.write("parameters\nb(i)/")
            num_rows_additional = bvip.shape[0]
            values_additional = [f"{row} {bvip[row, 0]}" for row in range(num_rows_additional)]
            file.write(", ".join(values_additional))
            file.write("/\n")
            num_rows_additional = raux.shape[0]
            file.write("r(l)/")
            values_additional = [f"{row} {raux[row, 0]}" for row in range(num_rows_additional)]
            file.write(", ".join(values_additional))
            file.write("/\n")
            file.write("s(l)/")
            values_additional = [f"{row} {saux[row, 0]}" for row in range(num_rows_additional)]
            file.write(", ".join(values_additional))
            file.write("/;\n\n")
            # Escrever o texto adicional ao final do arquivo
            file.write("positive variables\n")
            file.write("  yvip(k)  'variable of interest, aka decision variable'\n")
            file.write("  ;\n")
            file.write("equations\n")
            file.write("  F(i)   'FOC for agent optimization models'\n")
            file.write("  h   \n g \n")
            file.write(";\n")
            file.write("F(i)..  sum{k, A(i,k)*yvip(k)} + b(i) =N= 0;\n")
            file.write("h..  sum{l,r(l)*yvip(l)}=E= 1;\n")
            file.write("g..  sum{l,s(l)*yvip(l)**2}=L=" +str(FirmMax)+";\n")
            file.write("model VI / F, g, h /;\n")
            file.write("file annotations / '%emp.info%' /;\n")
            file.write("putclose annotations  'vi  F yvip g h'  ;\n")
            file.write("solve VI using emp;")
        
        jobvip=ws.add_job_from_file("output_table_VIP2.gms")
        ts0=time.time()
        jobvip.run()
        tprog=tprog+time.time()-ts0
        vvip = jobvip.out_db["yvip"]
        yvip=[]
        for rec in vvip:
            yvip.append(rec.level)
        yvip=np.array(yvip).reshape(2*P,1)
        ts1=time.time()
        xint=np.vstack((yvip[P:],xintvi,yvip[0:P]))
        #print(xint)
        xkn=(1-lamx)*xk+lamx*(xint+xk-yk/r)/2
        ykn=yk-lamy*r*(xint-xkn+(1/r)*yk)/2
        print(crit)
        crit=np.linalg.norm(xkn-xk)+np.linalg.norm(ykn-yk)
        #print(np.linalg.norm(ykn))
        xk=xkn
        yk=ykn
        tprog=tprog+time.time()-ts1
        print('iteração:', ite)
        ite=ite+1

    Tprcum.append(tprog)
    Totalitprog.append(ite)
    conta=conta+1
    print('Erro max=',np.linalg.norm(xM-xd)/np.linalg.norm(xd),np.linalg.norm(xk-xd)/np.linalg.norm(xd),max(abs(xd-xk)))
print('T_PATH=',Tdiretocum[1:],'T_DW=',Tdwcum[1:],'T_PROG=',Tprcum[1:])
print('Iteracoes: ','DW: ', Totalit[1:], 'Prog: ', Totalitprog[1:]) 
