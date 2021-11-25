function Export2XDMF(filename,TopologyType,Elems,Nodes,varargin)
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

if size(Nodes,2)==2
    Nodes = [Nodes,zeros(size(Nodes,1),1)];
end
FID = fopen([filename,'.xdmf'],'w');
fprintf(FID, ['<?xml version="1.0" ?>\n' ...
'<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n' ...
'<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n' ...
'<Domain Name="Domain%s">\n' ...
'<Grid Name="Grid%s" >\n' ...
'<Topology TopologyType="%s" NumberOfElements="%d"  >\n' ...
'<DataItem Format="XML" NumberType="int" Dimensions="%d %d">\n'],filename,filename,TopologyType,size(Elems,1), ...
    size(Elems,1),size(Elems,2));
formatstring = [repmat('%d ',1,size(Elems,2)),'\n'];
fprintf(FID,formatstring, Elems'-1);
fprintf(FID,'</DataItem>\n');
fprintf(FID,['</Topology>\n' ...
'<Geometry GeometryType="XYZ">\n' ...
'<DataItem Format="XML" NumberType="float" Precision="8" Dimensions="%d 3">\n'],size(Nodes,1));
fprintf(FID,'%12.8f %12.8f %12.8f \n', Nodes');
fprintf(FID,['</DataItem>\n' ...
'</Geometry>\n']);

for i=1:numel(varargin)
fprintf(FID,['<Attribute Name="%s" Center="%s" AttributeType="%s" >\n' ...
    '<DataItem Format="XML" NumberType="float" Precision="8" Dimensions="%d %d">\n'], ...
    varargin{i}.Name,varargin{i}.Center,varargin{i}.Type,size(varargin{i}.Values,1),size(varargin{i}.Values,2));
formatstring = [repmat('%12.8f ',1,size(varargin{i}.Values,2)),'\n'];
fprintf(FID,formatstring,varargin{i}.Values');
fprintf(FID,['</DataItem>\n' ...    
'</Attribute>\n']);
end
fprintf(FID,['</Grid>\n' ...
'</Domain>\n' ...
'</Xdmf>']);
fclose(FID);
end