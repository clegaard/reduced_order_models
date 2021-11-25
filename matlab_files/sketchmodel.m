function [Mesh,Pc,NumNodes,k,mask] = sketchmodel(h)


templatefile = 'MeshTemplate';

%% GMSH OPTIONS
options.gmsh = 'gmsh'; %GMSH
options.delete = false;
options.meshopts = '-1 -2 -order 1 -format msh2';
options.meshfile = 'DarcyBlockcoarse.msh';

%% Create and Import Mesh
Mesh = CreateGMSHfromTemplate(templatefile,options,'#h#',h);
% Mesh = fixMesh(Mesh);
%% Coordinates of the Inlet Boundary

Coord = GetCoord(Mesh,'Inlet');
CoordY=Coord(:,2);
[Y,Ind]=sort(CoordY);

%% Connectivity of the domain
CONECT{1}=GetConnectivity(Mesh,'Omega1');
CONECT{2}=GetConnectivity(Mesh,'Omega2');
CONECT{3}=GetConnectivity(Mesh,'Omega3');
CONECT{4}=GetConnectivity(Mesh,'Omega4');
CONECT{5}=GetConnectivity(Mesh,'Omega5');
CONECT{6}=GetConnectivity(Mesh,'Omega6');
CONECT{7}=GetConnectivity(Mesh,'Omega7');
CONECT{8}=GetConnectivity(Mesh,'Omega8');
CONECT{9}=GetConnectivity(Mesh,'Omega9');

%%% Connectivity of the boundaries
CInlet = GetConnectivity(Mesh,'Inlet');
CWall = GetConnectivity(Mesh,'Wall');
COutlet = GetConnectivity(Mesh,'Outlet');

%% Boundary Nodes

dofsDir = GetDofs(Mesh,'Outlet');
dofsNB1= GetDofs(Mesh,'Inlet');
dofsNB2= GetDofs(Mesh,'Wall');


ValNB1=ones(length(dofsNB1),1); % Neumann Boundary Value


%ValNB1=ones(length(dofsNB1),1); % Neumann Boundary Value

NumNodes= size(Mesh.X,1);
%% Permeability

k1=[CONECT{1}, ones(size(CONECT{1},1),1)];
k2=[CONECT{2}, ones(size(CONECT{2},1),1)]; 
k3=[CONECT{3}, ones(size(CONECT{3},1),1)];
k4=[CONECT{4}, ones(size(CONECT{4},1),1)];
k5=[CONECT{5}, 0.0005*ones(size(CONECT{5},1),1)];
k6=[CONECT{6}, ones(size(CONECT{6},1),1)];
k7= [CONECT{7}, ones(size(CONECT{7},1),1)];
k8=[CONECT{8}, ones(size(CONECT{8},1),1)];
k9= [CONECT{9}, ones(size(CONECT{9},1),1)];

k=[k1;k2;k3;k4;k5;k6;k7;k8;k9];

%% Assembly of stiffness and mass matrices and R.H.S vector
 mask=true(NumNodes,1);  
 mask(dofsDir)=false;    %%% Eliminate the nodes containing the Dirichlet Boundary Conditions
%%%% Assemble matrices for each sub domain 

 for i=1:9
     
     Kx{i} = Matrix2Dv5(Mesh.X(:,1:2),CONECT{i},1,0,1,0);
     Ky{i} = Matrix2Dv5(Mesh.X(:,1:2),CONECT{i},0,1,0,1);
     K{i}=Kx{i}+Ky{i};
     Pc.K{i}=K{i}(mask,mask);
     %M{i}= Matrix2Dv5(Mesh.X(:,1:2),CONECT{i},0,0,0,0);
     %P.M{i}=M{i}(mask,mask);
 end
 
 
 
 %% Mass Matrix on the inlet boundary
 M = FEM_mat_1D(Y,0,0);
 
 %% RHS vector
 f=zeros(NumNodes,1);
 aux=M*ValNB1;
 aux2=zeros(size(aux));
 aux2(Ind)= aux;
 f(dofsNB1)=aux2;  %% Apply Neumann BC
 Pc.rhs = f(mask) ;


end