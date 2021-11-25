function Connectivity = GetConnectivity(Mesh,DomainName)
f = find(strcmp(DomainName,Mesh.PHYSICAL_NAMES(:,3)));
dim =Mesh.PHYSICAL_NAMES{f,1};
tag =Mesh.PHYSICAL_NAMES{f,2};
Connectivity=Mesh.TOPO{dim}(Mesh.PHYSICAL_TAG{dim}==tag,:);
end
