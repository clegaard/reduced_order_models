clear all;
close all;
clc;

%%   Solving 2D steady state DarcyFlow (Diffusion) Problem
%DArcyBlock Problem with 9 different Permeabilities. 
%%(div.(-k/mu)(gradP))=0 , where k/mu is Permeability constant and P is pressure
addpath(genpath('./Libs'));
%% Mesh Generation using GMSH
hc = 0.5;  % Coarse Mesh size
hf = 0.1; % Fine Mesh size
permeability=[1,0.0005,1,1,0.0005,1,1,1,1];
%permeability=[1,1,1,1,1,1,1,1,1];
%% Assembly of the Hi fidelity Model

fprintf('Assembling the sketch model\n');
[CMesh,Pc,NumNodesc,kc,maskc] = sketchmodel(hc);  %Coarse Model
fprintf('Assembling the hi-fidelity model\n');
[FMesh,Pf,NumNodesf,kf,maskf] = hifi_model(hf) ;  % Fine Model
fprintf('Done!!\n');

%% Solve the Full order System
Pressure_c = DarcyBlockSolver(Pc,permeability);
Pressure_f = DarcyBlockSolver(Pf,permeability);

rhs_coarse = Pc.rhs;
rhs_fine(maskf) = Pf.rhs;

Kc = cell(9,1);
Kc_total = zeros(156,156); % TODO use NumNodesc - something

for i=1:9
    Kc{i}=Pc.K{i};
    Kc_total = Kc_total + Pc.K{i};
end

Kf = cell(9,1);
for i=1:9
    Kf{i} = zeros(NumNodesf);
    Kf{i}(maskf,maskf)=Pf.K{i};
end

%% Postprocessing the field variables
CCoord=GetCoord(CMesh,'All');
CConnectivity = GetConnectivity(CMesh,'All');
FCoord=GetCoord(FMesh,'All');
FConnectivity = GetConnectivity(FMesh,'All');

meshfile = 'CoarseMeshOnly';
Pressure_coarse=zeros(NumNodesc,1); 
Pressure_coarse(maskc)=Pressure_c;

%save('Pressure_c.mat','Pressure_coarse','CCoord')

Pressure_fine=zeros(NumNodesf,1); 
Pressure_fine(maskf)=Pressure_f;
trisurf( FConnectivity,FCoord(:,1), FCoord(:,2),Pressure_fine),axis equal,shading interp
xlabel('x')
ylabel('y')
zlabel('Pressure')

save('pressure.mat','Pressure_fine','Pressure_coarse','rhs_fine','rhs_coarse','FCoord','CCoord','Kc','Kf','permeability')

