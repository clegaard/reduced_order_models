function Coord = GetCoord(Mesh,DomainName)
C1 = GetConnectivity(Mesh,DomainName);
dofs = unique(C1(:));
Coord = Mesh.X(dofs,1:2);
end