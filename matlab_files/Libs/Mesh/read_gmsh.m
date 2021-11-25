function [X,TOPO,NUMBER_TAG,PHYSICAL_TAG,GEOMETRICAL_TAG,PHYSICAL_NAMES,elem_types,elem_names] = read_gmsh(filename)
    %GMSH mesh reader
    %Copyright (c) 2013, Adrien Leygue
    %Author: Adrien Leygue.
    %All rights reserved.
    %
    %INPUTS 
    % filename: a string identifying the file to read
    %
    %OUTPUTS
    % As a GMSH file can contain different element types the outputs will often
    % consist of cell arrays of length Nelem_types (Nelem_types is the number of different elements in the mesh)
    %
    % X the node coordinates Nnodes by 3 matrix where Nnodes is the number of nodes
    % TOPO a cell array of matrices describing the connectivity table of each element type
    % PHYSICAL_TAG a cell array of column vectors describing the physical tag of each element for each element type
    % GEOMETRICAL_TAG a cell array of column vectors describing the geometrical tag of each element for each element type
    % PHYSICAL_NAMES a cell array where each line gives the name associated to the different values of PHYSICAL_TAG
    % elem_types a line vector giving the number that GMSH associates to each element type ,see gmsh documentation
    % elem_names a cell array of strings providing a simple description of each element type,see gmsh documentation
    %
    %
    %
    %Redistribution and use in source and binary forms, with or without
    %modification, are permitted provided that the following conditions are met:
    %    * Redistributions of source code must retain the above copyright
    %      notice, this list of conditions and the following disclaimer.
    %    * Redistributions in binary form must reproduce the above copyright
    %      notice, this list of conditions and the following disclaimer in the
    %      documentation and/or other materials provided with the distribution.
    %    * The names of the owners and contributors of this software may not be used
    %      to endorse or promote products derived from this software without specific
    %      prior written permission.
    %
    %THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    %ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    %WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    %DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    %ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    %(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    %LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    %ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    %(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    %SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    %SYM CLASSES PACKAGE
    %
    %
    %Last modification: 25 oct. 2013
    
    assert(ischar(filename),'read_gmsh: argument 1 needs to be a string')
    fp = fopen(filename,'r');
    if(fp==-1)
        error(['Could not open file: ' filename]);
    end
    
    elem_types = [1 2 3 4 5 6 7 8 9 10 ...
                  11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24 25 26 27 28 29 30 ...
                  31 92  93];
    elem_names = {'EDGE_2', 'TRI_3', 'QUAD_4', 'TET_4', 'HEX_8', 'WEDGE_6', 'PYR_5', 'EDGE_3', 'TRI_6', 'QUAD_9',...
                  'TET_10', 'HEX_27', 'WEDGE_18', 'PYR_14', 'POINT', 'QUAD_8_i', 'HEX_20', 'WEDGE_15', 'PYR_13', 'TRI_9_i',...
                  'TRI_10', 'TRI_12_i','TRI_15','TRI_15_i', 'TRI_21', 'EDGE_4', 'EDGE_5', 'EDGE_6', 'TET_20', 'TET_35',...
                  'TET_56','HEX_64', 'HEX_125' };
    nodes_elem = [2 3 4 4 8 6 5 3 6 9  10 27 18 14  1  8 20 15 13  9 10 12 15 15 21  4  5  6 20 35 56 64 125];
    Ntypes = numel(elem_types);
    TOPO = cell(1,Ntypes);
    PHYSICAL_TAG = cell(1,Ntypes);
    GEOMETRICAL_TAG = cell(1,Ntypes);
    NUMBER_TAG = cell(1,Ntypes);
    PHYSICAL_NAMES = cell(0,3);
    
    
    
    while(~feof(fp))
        myline = fgetl(fp);
        %get rid of spaces
        myline = myline(regexp(myline,'[^ \f\n\r\t\v]'));
        switch myline
            case '$MeshFormat'
                myline = fgetl(fp);
                %display(['Version: ' myline]);
                myline = fgetl(fp);
                check_line(myline,'$EndMeshFormat');
                %Reads all the physical names
            case '$PhysicalNames'
                myline = fgetl(fp);
                Nitems = sscanf(myline,'%d',1);
                PHYSICAL_NAMES = cell(Nitems,3);
                for i=1:Nitems
                    myline = fgetl(fp);
                    data = sscanf(myline,'%d %d %*s',2);
                    PHYSICAL_NAMES{i,1} = data(1);
                    PHYSICAL_NAMES{i,2} = data(2);
                    PHYSICAL_NAMES{i,3} = char(sscanf(myline,'%*d %*d %s')');
                    PHYSICAL_NAMES{i,3}(PHYSICAL_NAMES{i,3}=='"') = '';
                end
                myline = fgetl(fp);
                check_line(myline,'$EndPhysicalNames');
                %Read all the nodes coordinates
            case {'$Nodes','$NOD'}
                myline = fgetl(fp);
                Nitems = sscanf(myline,'%d',1);
                X = fscanf(fp,'%*d %f %f %f',[3 Nitems]);
                X = X(1:3,:)';
                fgetl(fp); %jump to next line as fscanf will not jump line after reading the last node
                myline = fgetl(fp);
                check_line(myline,{'$EndNodes','$ENDNOD'});
                %Read all the elements
            case '$ELM'
                %This section is read twice. The first time the number of each
                %element types is determined
                myline = fgetl(fp);
                Nitems = sscanf(myline,'%d',1);
                %set a mark to rewind later   
                file_mark = ftell(fp);  
                %vector to accumulate the number of elements of each type
                Nitems_type = zeros(1,Ntypes);       
                for i=1:Nitems
                    myline = fgetl(fp);
                    loc_type = sscanf(myline,'%*d %d',1);
                    loc_pos = find(loc_type==elem_types);
                    Nitems_type(loc_pos) = Nitems_type(loc_pos) +1;
                end
                %Then memory is allocated
                for i=1:Ntypes
                    TOPO{i} = zeros(Nitems_type(i),nodes_elem(i));
                    PHYSICAL_TAG{i} = zeros(Nitems_type(i),1);
                    GEOMETRICAL_TAG{i} = zeros(Nitems_type(i),1);
                    NUMBER_TAG{i} = zeros(Nitems_type(i),1);
                end
                index_type = zeros(1,Ntypes);
                %rewind                
                fseek(fp,file_mark,'bof');
                % and the tables are filled during the second reading
                for i=1:Nitems
                    myline = fgetl(fp);
                    data = sscanf(myline,'%d');
                    loc_type = data(2);
                    loc_pos = find(loc_type==elem_types);
                    index_type(loc_pos) = index_type(loc_pos)+1;
                    NUMBER_TAG{loc_pos}(index_type(loc_pos)) = data(1);
                    PHYSICAL_TAG{loc_pos}(index_type(loc_pos)) = data(3);
                    GEOMETRICAL_TAG{loc_pos}(index_type(loc_pos)) = data(4);
                    TOPO{loc_pos}(index_type(loc_pos),:) = data(6:end);
                end
                myline = fgetl(fp);
                check_line(myline,'$ENDELM');
            case '$Elements'
                %This section is read twice. The first time the number of each
                %element types is determined
                myline = fgetl(fp);
                Nitems = sscanf(myline,'%d',1);
                %set a mark to rewind later
                file_mark = ftell(fp);
                %vector to accumulate the number of elements of each type
                Nitems_type = zeros(1,Ntypes);
                for i=1:Nitems
                    myline = fgetl(fp);
                    loc_type = sscanf(myline,'%*d %d',1);
                    loc_pos = find(loc_type==elem_types);
                    Nitems_type(loc_pos) = Nitems_type(loc_pos) +1;
                end
                %Then memory is allocated
                for i=1:Ntypes
                    TOPO{i} = zeros(Nitems_type(i),nodes_elem(i));
                    PHYSICAL_TAG{i} = zeros(Nitems_type(i),1);
                    GEOMETRICAL_TAG{i} = zeros(Nitems_type(i),1);
                    NUMBER_TAG{i} = zeros(Nitems_type(i),1);
                end
                index_type = zeros(1,Ntypes);
                %rewind
                fseek(fp,file_mark,'bof');
                % and the tables are filled during the second reading
                for i=1:Nitems
                    myline = fgetl(fp);
                    data = sscanf(myline,'%d');
                    loc_type = data(2);
                    loc_pos = find(loc_type==elem_types);
                    index_type(loc_pos) = index_type(loc_pos)+1;
                    NUMBER_TAG{loc_pos}(index_type(loc_pos)) = data(1);
                    if(data(3)>=1)
                        PHYSICAL_TAG{loc_pos}(index_type(loc_pos)) = data(4);
                    end
                    if(data(3)>=2)
                        GEOMETRICAL_TAG{loc_pos}(index_type(loc_pos)) = data(5);
                    end
                    TOPO{loc_pos}(index_type(loc_pos),:) = data(4+data(3):end);
                end
                myline = fgetl(fp);
                check_line(myline,'$EndElements');
            otherwise
                error(['Unknown section: ' myline]);
        end
    end
    mask = ~cellfun(@isempty,TOPO); %isempty does not work as we have arrays with 0 lines and several columns
    TOPO = TOPO(mask);
    NUMBER_TAG = NUMBER_TAG(mask);
    PHYSICAL_TAG = PHYSICAL_TAG(mask);
    GEOMETRICAL_TAG = GEOMETRICAL_TAG(mask);
    elem_types = elem_types(mask);
    elem_names = elem_names(mask);
    fclose(fp);
end

%Utility function... for historical reasond
function check_line(l,s)
    l = l(regexp(l,'[^ \f\n\r\t\v]'));
    if(~any(strcmp(l,s)))
        error(['different strings:' l ' ' s]);
    end
end