function dofs = GetDofs(Mesh,DomainName)
C1 = GetConnectivity(Mesh,DomainName);
dofs = unique(C1(:));
end
