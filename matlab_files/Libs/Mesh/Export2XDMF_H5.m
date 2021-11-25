function Export2XDMF_H5(filename,TopologyType,Elems,Nodes,varargin)
% Export the file filename.xdmf
% TopologyType . One of the following:
%% Linear
% Polyvertex - a group of unconnected points
% Polyline - a group of line segments
% Polygon
% Triangle
% Quadrilateral
% Tetrahedron
% Pyramid
% Wedge
% Hexahedron
%% Quadratic
% Edge_3 - Quadratic line with 3 nodes
% Tri_6
% Quad_8
% Tet_10
% Pyramid_13
% Wedge_15
% Hex_20
%% Arbitrary
% Mixed - a mixture of unstructured cells
%% Structured
% 2DSMesh - Curvilinear
% 2DRectMesh - Axis are perpendicular
% 2DCoRectMesh - Axis are perpendicular and spacing is constant
% 3DSMesh
% 3DRectMesh
% 3DCoRectMesh
% Elems : Connectivity Table
% Nodes : List of nodes 
% varargin : Fields to export: 
% Each field has to have the following properties:
% 
%Field.Name     : a string with the name
%Field.Center   : 'Node', 'Edge','Face','Cell','Grid'
%Field.Type     : Scalar | Vector | Tensor | Tensor6 | Matrix | GlobalID
%Field.Values   : An array containing the numerical values


% Isolate the name from the directory path
ind = find(filename == '/');
if any(ind)
    name = filename(ind(end)+1:end);
else
    name = filename;
end

if size(Nodes,2)==2
    GeometryType='XY';
else
    GeometryType='XYZ';
end
if exist([filename,'.h5'],'file')
    delete([filename,'.h5']); % delete old file it already exists
end
h5create([filename,'.h5'],'/Elements',size(Elems'),'Datatype','int64');
h5write([filename,'.h5'],'/Elements',int64(Elems-1)');
h5create([filename,'.h5'],'/Nodes',size(Nodes'),'Datatype','double');
h5write([filename,'.h5'],'/Nodes',Nodes');

FID = fopen([filename,'.xdmf'],'w');
fprintf(FID, ['<?xml version="1.0" ?>\n' ...
'<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n' ...
'<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n' ...
'<Domain Name="Domain%s">\n' ...
'<Grid Name="Grid%s" >\n' ...
'<Topology TopologyType="%s" NumberOfElements="%d"  >\n' ...
'<DataItem Format="HDF" NumberType="int" Dimensions="%d %d">\n'],name,name,TopologyType,size(Elems,1), ...
    size(Elems,1),size(Elems,2));
fprintf(FID,[name,'.h5:/Elements\n']);
fprintf(FID,'</DataItem>\n');
fprintf(FID,['</Topology>\n' ...
'<Geometry GeometryType="%s">\n' ...
'<DataItem Format="HDF" NumberType="float" Precision="8" Dimensions="%d %d">\n'],GeometryType,size(Nodes,1),size(Nodes,2));
fprintf(FID,[name,'.h5:/Nodes\n']);
fprintf(FID,['</DataItem>\n' ...
'</Geometry>\n']);

for i=1:numel(varargin)
h5create([filename,'.h5'],['/',varargin{i}.Name],size(varargin{i}.Values'),'Datatype','double');
h5write([filename,'.h5'],['/',varargin{i}.Name],varargin{i}.Values');
fprintf(FID,['<Attribute Name="%s" Center="%s" AttributeType="%s" >\n' ...
    '<DataItem Format="HDF" NumberType="float" Precision="8" Dimensions="%d %d">\n'], ...
    varargin{i}.Name,varargin{i}.Center,varargin{i}.Type,size(varargin{i}.Values,1),size(varargin{i}.Values,2));
fprintf(FID,[name,'.h5:/',varargin{i}.Name,'\n']);
fprintf(FID,['</DataItem>\n' ...    
'</Attribute>\n']);
end
fprintf(FID,['</Grid>\n' ...
'</Domain>\n' ...
'</Xdmf>']);
fclose(FID);
end