
%error('Tracker not configured! Please edit the tracker_DRNet.m file.'); % Remove this line after proper configuration

% The human readable label for the tracker, used to identify the tracker in reports
% If not set, it will be set to the same value as the identifier.
% It does not have to be unique, but it is best that it is.
tracker_label = ['DRNet'];

CODE_PATH = '/home/baishuai/experiment/pytracking/DRNet/'; %set the code path
TRAX_BUILD_PARH = '/home/baishuai/experiment/trax/build/'; %set the trax build path

setenv('MKL_NUM_THREADS','1')
tracker_command = generate_python_command('run_DRNet', {CODE_PATH, [CODE_PATH 'pytracking/VOT2019'],[TRAX_BUILD_PARH 'python/lib']});
tracker_interpreter = 'python';

tracker_linkpath = {TRAX_BUILD_PARH}; % A cell array of custom library directories used by the tracker executable (optional)

