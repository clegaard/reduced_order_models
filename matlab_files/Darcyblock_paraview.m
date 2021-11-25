clear all;
close all;
clc;
%%
addpath(genpath('./Libs'));
%% Solving 2D steady state DarcyFlow (Diffusion) Problem
%%DarcyBlock Problem with 9 different Permeabilities. 
%%(div.(-k/mu)(gradP))=0 , where k/mu is Permeability constant and P is pressure
% Snapshot generation for each Permeability parameter(9 dimensions) by
% Successive Model Sketching
%Permeability Sampling done by Latin HyperCube 

%% Mesh Generation using GMSH 
% create mesh from template 
hc = 0.5;  % Coarse Mesh size
hf = 0.1; % Fine Mesh size

%% Permeability Sampling
r1= -5;
r2 = 0;
 
Ns = 1000 ;
Ntest = 50;
 
Sampling = 10.^(lhsu(r1*ones(9,1),r2*ones(9,1),Ns+Ntest));
Permeability=Sampling(1:Ns,:);
Permeability_new=Sampling(Ns+1:end,:);

%%   Solving 2D steady state DarcyFlow (Diffusion) Problem
%% Assembly of the Hi fidelity Model

fprintf('Assembling the sketch model\n');
[CMesh,Pc,NumNodesc,kc,maskc] = sketchmodel(hc);  %Coarse Model
fprintf('Assembling the hi-fidelity model\n');
[FMesh,Pf,NumNodesf,kf,maskf] = hifi_model(hf) ;  % Fine Model
fprintf('Done!!\n');

%% Solve the Full order System
Pressure_c = DarcyBlockSolver(Pc,Permeability);
Pressure_f = DarcyBlockSolver(Pf,Permeability);

%% Postprocessing the field variables
CCoord=GetCoord(CMesh,'All');
CConnectivity = GetConnectivity(CMesh,'All');
FCoord=GetCoord(FMesh,'All');
FConnectivity = GetConnectivity(FMesh,'All');

meshfile = 'CoarseMeshOnly';
Pressure_coarse=zeros(NumNodesc,size(Permeability,1)); 
Pressure_coarse(maskc,:)=Pressure_c;

%save('Pressure_c.mat','Pressure_coarse')

Pressure_fine=zeros(NumNodesf,size(Permeability,1)); 
Pressure_fine(maskf,:)=Pressure_f;
trisurf( FConnectivity,FCoord(:,1), FCoord(:,2),Pressure_fine(:,1)),axis equal,shading interp
xlabel('x')
ylabel('y')
zlabel('Pressure')
%save('Pressure_f.mat','Pressure_fine')

Field_1.Name  = 'Pressure';
Field_1.Center = 'Node';
Field_1.Type   = 'Scalar';
Field_1.Values =  Pressure_coarse;
 
Field_2.Name  = 'Permeability';
Field_2.Center = 'Cell';
Field_2.Type   = 'Scalar';
Field_2.Values = kc(:,4);

 
Export2XDMF_H5(meshfile,'Triangle',CConnectivity,CMesh.X,Field_1,Field_2);
[k,om]=parameter(CCoord); 
kk=[k{1};k{2};k{3};k{4};k{5};k{6};k{7};k{8};k{9}];
triplot(CConnectivity,CCoord(:,1),CCoord(:,2))
[c,ia,ic]=unique(kk(:,1),'rows','stable');
kkk=kk(ia,:);
trisurf(CConnectivity,CCoord(:,1),CCoord(:,2),kkk(:,2))
[xx,yy]=meshgrid(CCoord(:,1),CCoord(:,2));
ll=reshape(kkk(:,2),xx,yy);
%surf(CCoord(:,1),CCoord(:,2),c)

%% ICITECH mesh import
distributed=false;
numelem_coarse=size(CConnectivity,1);
elements_coarse=[CConnectivity(:,1) zeros(numelem_coarse,1) CConnectivity(:,2) zeros(numelem_coarse,1)  CConnectivity(:,3) zeros(numelem_coarse,1)];
nnodes_coarse=size(CCoord,1);
dofs=3;
infoc=[nnodes_coarse 2 numelem_coarse dofs];
dlmwrite('coarse00_0.T',distributed)
dlmwrite('coarse00_0.T',infoc,'-append','delimiter','\t');
dlmwrite('coarse00_0.T',CCoord,'-append','delimiter','','precision','%10.4f')
dlmwrite('coarse00_0.T', elements_coarse, '-append','delimiter', '\t')

numelem_fine=size(FConnectivity,1);
elements_fine=[FConnectivity(:,1) zeros(numelem_fine,1) FConnectivity(:,2) zeros(numelem_fine,1)  FConnectivity(:,3) zeros(numelem_fine,1)];
nnodes_fine=size(FCoord,1);
infof=[nnodes_fine 2 numelem_fine dofs];
dlmwrite('fine00_0.T',distributed)
dlmwrite('fine00_0.T',infof,'-append','delimiter','\t');
dlmwrite('fine00_0.T',FCoord,'-append','delimiter','','precision','%10.4f')
dlmwrite('fine00_0.T', elements_fine, '-append','delimiter', '\t')

%%
order=num2str(1);
continuous='C';
ncomp=num2str(1);
order=[order continuous ncomp];
nval_coarse=[0 nnodes_coarse 0 0 0 0];
nval_fine=[0 nnodes_fine 0 0 0 0];

for j=1
filename = sprintf('coarse00_pk%d_0.F',j);
dlmwrite(filename,distributed)
dlmwrite(filename,order,'-append','delimiter','\t');
dlmwrite(filename,nval_coarse,'-append','delimiter','\t');
dlmwrite(filename,Pressure_coarse(:,j),'-append','precision','%10.4f')
end

for j=1
filename = sprintf('fine00_pk%d_0.F',j);
dlmwrite(filename,distributed)
dlmwrite(filename,order,'-append','delimiter','\t');
dlmwrite(filename,nval_fine,'-append','delimiter','\t');
dlmwrite(filename,Pressure_fine(:,j),'-append','delimiter','\t','precision','%10.4f')
end