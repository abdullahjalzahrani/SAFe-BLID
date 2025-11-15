function [X, Y, feature_names] = load_dataset(name, folder)

fname = fullfile(folder, [name '.csv']);
if ~exist(fname,'file')
    error('Dataset file not found: %s', fname);
end

T = readtable(fname);

feature_names = T.Properties.VariableNames(1:end-1);
X = table2array(T(:,1:end-1));
Y = table2array(T(:,end));

end
