function A= Matrix2Dv5(COORD,CONECT,op1x,op1y,op2x,op2y,lin_fields,semilin_x,semilin_y,cst_fields)

% GENERALIZED FOR NON-UNIFORM MESHES. CODE ALREADY VERIFIED.

% ACCEPTS SEMI-LINEAR FIELDS: 
% EX. grad(v)*grad(c)*u, where v is the test function, u is the unknown and 
% c is a known field (linear elements supposed). 
% Then, grad(c) = dc/dx.êx + dc/dy.êy. The term dc/dx produces a 
% constant function in the x-direction but a linear one in the y-direction,
% and on the contrary for dc/dy.

if nargin < 6
    error('At least 6 input arguments must be provided')
elseif nargin > 11
    error('No more than 11 input arguments may be provided')
elseif nargin == 6
    lin_fields = [];
    semilin_x = [];
    semilin_y = [];
    cst_fields = [];
elseif nargin == 7
    semilin_x = [];
    semilin_y = [];
    cst_fields = [];
elseif nargin == 8
    semilin_y = [];
    cst_fields = [];
elseif nargin == 9
    cst_fields = [];
end

if size(CONECT,2) == 3
    shape = 'tri';
elseif size(CONECT,2) == 4
    shape = 'quad';
else
    error('Connectivity matrix provided has not the appropiate format')
end

Nnodes=size(COORD,1);
Nelem=size(CONECT,1);

Shap=cell(3,1);
switch shape
    case 'quad'
        Shap{1} = @(xi,yi,Xe) [(1-xi).*(1-yi)/4;(1+xi).*(1-yi)/4;(1+xi).*(1+yi)/4;(1-xi).*(1+yi)/4];
        Dx = @(xi,yi,Xe) [(yi-1)/4;(1-yi)/4;(1+yi)/4;-(1+yi)/4];
        Dy = @(xi,yi,Xe) [(xi-1)/4;-(xi+1)/4;(xi+1)/4;(1-xi)/4];
        Lin_eval = @(xi,yi,Xe) ((1-xi).*(1-yi)*Xe(1)+(1+xi).*(1-yi)*Xe(2)+(1+xi).*(1+yi)*Xe(3)+(1-xi).*(1+yi)*Xe(4))/4;
        Cst_eval = @(xi,yi,Xe) Xe*ones(1,length(xi));
        Slinx_eval = @(xi,yi,Xe) ((-1)*(1-yi)*Xe(1)+(1-yi)*Xe(2)+(1+yi)*Xe(3)+(-1)*(1+yi)*Xe(4))/4;
        Sliny_eval = @(xi,yi,Xe) ((1-xi)*(-1)*Xe(1)+(1+xi)*(-1)*Xe(2)+(1+xi)*Xe(3)+(1-xi)*Xe(4))/4;
        Jac = @(xi,yi,Xe,Ye) [(yi-1)/4 (1-yi)/4 (1+yi)/4 -(1+yi)/4;(xi-1)/4 -(1+xi)/4 (1+xi)/4 (1-xi)/4]*[Xe(1) Ye(1);Xe(2) Ye(2);Xe(3) Ye(3);Xe(4) Ye(4)];
        xi_integ = [-0.577350269189626 0.577350269189626 0.577350269189626 -0.577350269189626];
        yi_integ = [-0.577350269189626 -0.577350269189626 0.577350269189626 0.577350269189626];
        w_integ = [1 1 1 1];
        vertices = [-1 -1;1 -1;1 1;-1 1];
    case 'tri'
        Shap{1}=@(xi,yi,Xe) [1-xi-yi;xi;yi];
        Dx=@(xi,yi,Xe) [-ones(1,numel(xi));ones(1,numel(xi));zeros(1,numel(xi))];
        Dy=@(xi,yi,Xe) [-ones(1,numel(yi));zeros(1,numel(yi));ones(1,numel(yi))];
        Lin_eval = @(xi,yi,Xe) (1-xi-yi)*Xe(1)+xi*Xe(2)+yi*Xe(3);
        Cst_eval = @(xi,yi,Xe) Xe*ones(1,length(xi));
        Slinx_eval = @(xi,yi,Xe) (-yi)*Xe(1)+ones(length(xi),1)*Xe(2)+zeros(length(yi),1)*Xe(3);
        Sliny_eval = @(xi,yi,Xe) (-xi)*Xe(1)+zeros(length(xi),1)*Xe(2)+ones(length(yi),1)*Xe(3);
        Jac = @(xi,yi,Xe,Ye) [-1 1 0;-1 0 1]*[Xe(1) Ye(1);Xe(2) Ye(2);Xe(3) Ye(3)];
        xi_integ = [0.5 0 0.5];%1/3;
        yi_integ = [0 0.5 0.5];%1/3;
        w_integ = [1/6 1/6 1/6];%1/2;
    otherwise
end

A = sparse(Nnodes,Nnodes);
    for i=1:Nelem
        map = CONECT(i,:);
        Xelem = COORD(map,1);
        Yelem = COORD(map,2);
        
        loc_jac=zeros(1,length(w_integ));
        Shap{2}=[];
        Shap{3}=[];
        
        for k=1:length(w_integ)
            JAC = Jac(xi_integ(k),yi_integ(k),Xelem,Yelem);
            loc_jac(k)=det(JAC);
            DERIV=[Dx(xi_integ(k),yi_integ(k),Xelem) Dy(xi_integ(k),yi_integ(k),Xelem)]';
            InvJAC=(1/loc_jac(k))*[JAC(2,2) -JAC(1,2);-JAC(2,1) JAC(1,1)];
            DERIV=InvJAC*DERIV;        
            Shap{2}=[Shap{2};DERIV(1,:)];
            Shap{3}=[Shap{3};DERIV(2,:)];
        end
        
        if (op1x==0 && op1y==0)
            M1 = Shap{1}(xi_integ,yi_integ,Xelem);
        else 
            M1= Shap{1+op1x+2*op1y}';
        end
        
        if (op2x==0 && op2y==0)
            M2 = Shap{1}(xi_integ,yi_integ,Xelem);
        else
            M2 = Shap{1+op2x+2*op2y}';
        end
        
        D = w_integ;
        for j=1:size(lin_fields,2)
            D = D.*Lin_eval(xi_integ,yi_integ,lin_fields(map,j));
        end      
        for j=1:size(cst_fields,2)
            D = D.*Cst_eval(xi_integ,yi_integ,cst_fields(i,j));
        end
        for j=1:size(semilin_x,2)
            aux=Slinx_eval(xi_integ,yi_integ,semilin_x(map,j));
            for k=1:length(aux)
                nodJac=Jac(vertices(k,1),vertices(k,2),Xelem,Yelem);
                InvnodJac=(1/det(nodJac))*[nodJac(2,2) -nodJac(1,2);-nodJac(2,1) nodJac(1,1)];
                D = D.*aux(k)*InvnodJac(1,1);
            end
        end
        for j=1:size(semilin_y,2)
            aux=Sliny_eval(xi_integ,yi_integ,semilin_y(map,j));
            for k=1:length(aux)
                nodJac=Jac(vertices(k,1),vertices(k,2),Xelem,Yelem);
                InvnodJac=(1/det(nodJac))*[nodJac(2,2) -nodJac(1,2);-nodJac(2,1) nodJac(1,1)];
                D = D.*aux(k)*InvnodJac(2,2);
            end
        end
        D=D.*loc_jac;
        
        A(map,map) = A(map,map)+(M1*diag(D)*M2');       
        
    end

end