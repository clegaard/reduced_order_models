%------------------------------------------------------------------------%
%------ Gmsh to Matlab script: Import mesh to matlab---------------------%
%------------------------------------------------------------------------%

clc
close all
clear 

%-----------------------------------------------------------------------%
% dlmread(filename,delimiter,[R1 C1 R2 C2]) reads only the range 
% bounded by row offsets R1 and R2 and column offsets C1 and C2.
%-----------------------------------------------------------------------%
addpath(genpath('../src/'))
addpath(genpath('../../Libs/'))
path2data='../../data_fine_normalised/';
P=load('Matrices');
Mesh = GetTRI3fromTRI6(P.Mesh);
file    =  ('output.msh');

% no of nodes is mentioned in 5th row and first column

N_n      = dlmread(file,'',[14-1 1-1 14-1 1-1]);
N_e      = dlmread(file,'',[16+N_n 0 16+N_n 0]);
distributed=false;
%node_id     = dlmread(file,'',[14 0 13+N_n 0]);
%nodes_vel       = dlmread(file,'',[14 1 13+N_n 3]);
%elements_vel    = dlmread(file,'',[17+N_n 0 16+N_n+N_e 7]);

nodes_conf=Mesh.X(:,1:2);
elements_conf=[Mesh.TOPO{2}(:,1) zeros(size(Mesh.TOPO{2},1),1) Mesh.TOPO{2}(:,2) zeros(size(Mesh.TOPO{2},1),1)  Mesh.TOPO{2}(:,3)];
nnodes=size(Mesh.X(:,1),1);
nelements=size(Mesh.TOPO{2},1);
dofs=3;
info=[nnodes 2 nelements dofs];
dlmwrite('myFile.T',distributed)
dlmwrite('myFile.T',info,'-append','delimiter','\t');
dlmwrite('myFile.T',nodes_conf,'-append','delimiter','','precision','%10.4f')
dlmwrite('myfile.T', elements_conf, '-append','delimiter', '\t')

order=num2str(2);
continuous='C';
ncomp=num2str(3);


load([path2data,'LogC_FM.mat']);
sol=LogC_FM;
ssol=size(sol{1},1)/3;
% logc_zz=zeros(numel(P.tspan)*ssol,1);
% logc_rr=zeros(numel(P.tspan)*ssol,1);
% logc_zr=zeros(numel(P.tspan)*ssol,1);
% 
%  for j=1:10
%  logc_zz = reshape(sol{j}(1:ssol,:),[],1);
%  logc_rr = reshape(sol{j}(ssol+1:2*ssol,:),[],1);
%  logc_zr = reshape(sol{j}(2*ssol+1:end,:),[],1);
%  logc{j}=[logc_zz logc_rr logc_zr];
%  end
 
logc_zz=zeros(ssol,numel(P.tspan));
logc_rr=zeros(ssol,numel(P.tspan));
logc_zr=zeros(ssol,numel(P.tspan));

for j=1:10
logc_zz = sol{j}(1:ssol,:);
logc_rr = sol{j}(ssol+1:2*ssol,:);
logc_zr =sol{j}(2*ssol+1:end,:);
   for i=1:numel(P.tspan)
   logc{j}{i}=[logc_zz(:,i) logc_rr(:,i) logc_zr(:,i)];
   end
 end
 
 
 
order=[order continuous ncomp];
nval=[0 nnodes 0 0 0 0];

for j=1:10
for i=1:100
 filename = sprintf('solFile_mu%d_time%d.F',j,i);
dlmwrite(filename,distributed)
dlmwrite(filename,order,'-append','delimiter','\t');
dlmwrite(filename,nval,'-append','delimiter','\t');
dlmwrite(filename,logc{j}{i},'-append','delimiter','\t','precision','%10.4f')
end
end 
