function varargout=CreateGMSHfromTemplate(templatefile,options,varargin)
% Function Create New Mesh from template file 'templatefile' using new
% parameters. The code replaces the parameters '#par1#','#par2#' with the
% respective values.
% ex: 
%M=CreateGMSHfromTemplate('template.geo',options,'#par1#',0.1,'#par2#',10);
% options are the options used for meshing under GMSH
[filepath,name,ext] = fileparts(templatefile);
if isempty(filepath)
    filepath = '.';
end
if isempty(ext)
    ext = '.geo';
end
newgeo = [filepath,'/',name,'_aux',ext];
meshfile = [filepath,'/',options.meshfile];
filetext = fileread(templatefile);
old = varargin(1:2:end);
new = cellfun(@num2str,varargin(2:2:end),'UniformOutput',false);
newtext = replace(filetext,old,new);
fid = fopen(newgeo,'wt');
fprintf(fid, newtext);
fclose(fid);
command = [options.gmsh,' ',newgeo,' ',options.meshopts,' -o ',meshfile];
status = system(command);
if nargout>0    
    [varargout{1}.X,varargout{1}.TOPO,varargout{1}.NUMBER_TAG,varargout{1}.PHYSICAL_TAG,varargout{1}.GEOMETRICAL_TAG,varargout{1}.PHYSICAL_NAMES,varargout{1}.elem_types,varargout{1}.elem_names] = read_gmsh(meshfile);
    if options.delete
        delete(newgeo,meshfile);
    end    
end
end