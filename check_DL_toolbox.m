% small helper
function tf = check_DL_toolbox()
    % check for modern deep learning functions
    tf = exist('trainNetwork','file')==2 && exist('sequenceInputLayer','file')==2 && exist('bilstmLayer','file')==2;
end