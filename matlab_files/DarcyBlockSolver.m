function Pressure = DarcyBlockSolver(P,Permeability)

Pressure = zeros(size(P.K{1},1),size(Permeability,1));

for k=1:size(Permeability,1)
     K=Permeability(k,1).*P.K{1};
     for j= 2:9
     K= K+Permeability(k,j).*P.K{j};
     end
    %rhs= Permeability(k,4)*P.rhs; % constant pressure gradient
    rhs= P.rhs; % constant inlet velocity
    Pressure(:,k) = K\rhs; 
end


end

